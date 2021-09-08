import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
from models.parsing_model import BiSeNet
from torch.nn import functional as F
import torch.nn as nn

class GESGModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_par', type=float, default=0.4, help='weight for semantic loss')
            parser.add_argument('--lambda_gra', type=float, default=50, help='weight for gradient loss')
            parser.add_argument('--lambda_gra_L1', type=float, default=0.5, help='weight for pixel gradient loss')
            parser.add_argument('--lambda_D1', type=float, default=0.1, help='weight for D_sm')
            parser.add_argument('--lambda_D2', type=float, default=0.4, help='weight for D_sm')
            parser.add_argument('--lambda_D3', type=float, default=0.4, help='weight for D_sm')
            parser.add_argument('--lambda_part_adv', type=float, default=0.1, help='weight for semantic adversarial loss')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = [ 'G_A', 'G_par','cycle_A', 'idt_A','par_A', 'D_B', 'G_B', 'cycle_B', 'idt_B','par_B','gra','gra_L1','D_A_all']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['fake_B']
        visual_names_B = ['real_B']

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B','G_gra','D_1','D_2','D_3']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B', 'G_gra']

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'resnet_9blocks_A', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'resnet_9blocks', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_gra = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'resnet_9blocks_gra', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.parsing_net = BiSeNet(19)
        self.parsing_net.cuda()
        self.parsing_net.load_state_dict(torch.load('models/79999_iter.pth')) #pretrained parsing model
        self.parsing_net.eval()

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_1 = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_2 = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            1, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_3 = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            1, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters() ,self.netG_gra.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), self.netD_1.parameters(), self.netD_2.parameters(), self.netD_3.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def cal_par(self,real):
        # calculate parsing maps of images
        real512 = F.interpolate(real, scale_factor=2, mode='nearest')

        tmp = self.parsing_net(real512)[0]
        tmp_par = torch.tensor(torch.max(tmp, 1)[1], dtype=torch.float)

        tmp_par2 = torch.unsqueeze(tmp_par, 0)
        par256 = F.interpolate(tmp_par2, scale_factor=0.5, mode='nearest').cuda()
        return par256

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.graimg_real_A = self.cal_gra(self.real_A)
        self.graimg_real_B = self.cal_gra(self.real_B)
        self.grasyn_real_A, self.graft_real_A, self.ft_list_real_A = self.netG_gra(self.graimg_real_A)
        self.grasyn_real_B, self.graft_real_B, self.ft_list_real_B = self.netG_gra(self.graimg_real_B)

        self.par_A = self.cal_par(self.real_A)
        self.par_B = self.cal_par(self.real_B)
        self.majormask_realA = self.cal_majormask(self.par_A)
        self.majormask_realB = self.cal_majormask(self.par_B)
        self.fake_B = self.netG_A(self.real_A, self.graft_real_A, self.majormask_realA, self.ft_list_real_A)  # G_A(A)

        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)

        self.par_A_fake = self.cal_par(self.fake_A)
        self.par_B_fake = self.cal_par(self.fake_B)
        self.majormask_fakeA = self.cal_majormask(self.par_A_fake)

        self.cal_real_fake_par()

        self.graimg_fake_A = self.cal_gra(self.fake_A)
        self.grasyn_fake_A, self.graft_fake_A, self.ft_list_fake_A= self.netG_gra(self.graimg_fake_A)
        self.rec_B = self.netG_A(self.fake_A,self.graft_fake_A,self.majormask_fakeA, self.ft_list_fake_A)

        self.graavg_real_B = self.cal_graavg(self.graimg_real_B)
        self.graavg_syn = self.cal_graavg(self.grasyn_real_A)


    def cal_gra(self,x):
        # calculate gradient maps of images
        gradient_model = Gradient_Net().to(self.device)
        g = gradient_model(x)
        return g

    def cal_graavg(self,g):
        # avgpool and reshape gradient maps
        avg_pool = torch.nn.AvgPool2d(3, 2)
        g_avg = avg_pool(g)
        avg0 = g_avg[:, 0].view(1, g_avg.shape[2] * g_avg.shape[3])
        avg1 = g_avg[:, 1].view(1, g_avg.shape[2] * g_avg.shape[3])
        avg2 = g_avg[:, 2].view(1, g_avg.shape[2] * g_avg.shape[3])
        g_res = torch.zeros([3, g_avg.shape[2] * g_avg.shape[3]])
        g_res[0] = avg0
        g_res[1] = avg1
        g_res[2] = avg2
        self.normhis = float(g_avg.shape[2] * g_avg.shape[3])
        return g_res


    def cal_his(self,x, y):
        #calculate gradient histogram from the reshaped maps
        lambda_aug = 8
        x = x * lambda_aug
        x = x.floor()
        x = x.long()

        y = y * lambda_aug
        y = y.floor()
        y = y.long()

        his_x = torch.bincount(x)
        his_y = torch.bincount(y)

        if his_x.size()[0] > his_y.size()[0]:
            length = his_x.size()[0] - his_y.size()[0]
            new_arr = torch.zeros(length)
            arr3 = torch.cat([his_y.float(), new_arr], 0)
            loss = self.criterionL1(arr3, his_x.float())
        elif his_x.size()[0] < his_y.size()[0]:
            length = his_y.size()[0] - his_x.size()[0]
            new_arr = torch.zeros(length)
            arr3 = torch.cat([his_x.float(), new_arr], 0)
            loss = self.criterionL1(arr3, his_y.float())
        else:
            loss = self.criterionL1(his_y.float(), his_x.float())
        return loss /self.normhis

    def cal_real_fake_par(self):
        #seperate different facial component of real images and fake images.
        arr1 = torch.zeros(256, 256).cuda() + 1
        arr0 = torch.zeros(256, 256).cuda()

        mask1 = torch.where(torch.logical_or(torch.logical_or(torch.logical_or(self.par_B == 2, self.par_B == 3),
                                                              torch.logical_or(self.par_B == 4, self.par_B == 5)),
                                             torch.logical_or(torch.logical_or(self.par_B == 12, self.par_B == 13),
                                                              self.par_B == 10))
                            , arr1, arr0)
        mask2 = torch.where(self.par_B == 1, arr1, arr0)
        mask3 = torch.where(
            torch.logical_or(torch.logical_or(self.par_B == 0, torch.logical_and(self.par_B > 5, self.par_B < 10))
                             , torch.logical_and(self.par_B > 13, self.par_B < 19))
            , arr1, arr0)

        mask1_f = torch.where(
            torch.logical_or(torch.logical_or(torch.logical_or(self.par_B_fake == 2, self.par_B_fake == 3),
                                              torch.logical_or(self.par_B_fake == 4, self.par_B_fake == 5)),
                             torch.logical_or(torch.logical_or(self.par_B_fake == 12, self.par_B_fake == 13),
                                              self.par_B_fake == 10))
            , arr1, arr0)
        mask2_f = torch.where(self.par_B_fake == 1, arr1, arr0)
        mask3_f = torch.where(
            torch.logical_or(
                torch.logical_or(self.par_B_fake == 0, torch.logical_and(self.par_B_fake > 5, self.par_B_fake < 10))
                , torch.logical_and(self.par_B_fake > 13, self.par_B_fake < 19))
            , arr1, arr0)

        self.real1 = mask1.cuda() * self.real_B
        self.real2 = mask2.cuda() * self.real_B
        self.real3 = mask3.cuda() * self.real_B

        self.fake1 = mask1_f.cuda() * self.fake_B
        self.fake2 = mask2_f.cuda() * self.fake_B
        self.fake3 = mask3_f.cuda() * self.fake_B



    def cal_majormask(self,par_A):
        # separate parsing masks into several major masks
        arr1 = torch.zeros(256, 256).cuda() + 1
        arr0 = torch.zeros(256, 256).cuda()

        mask1 = torch.where(torch.logical_or(torch.logical_or(torch.logical_or(par_A == 2, par_A == 3),
                                                              torch.logical_or(par_A == 4, par_A == 5)),
                                             torch.logical_or(par_A == 12, par_A == 13)), arr1, arr0)#comp
        mask2 = torch.where(par_A==17, arr1, arr0)#hair
        mask3 = torch.where(par_A==10, arr1, arr0) #nose
        mask1 = mask1.cuda()
        mask2 = mask2.cuda()
        mask3 = mask3.cuda()
        mask =[mask1,mask2,mask3]

        return mask


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def backward_D_par(self):
        self.loss_D_1 = self.backward_D_basic(self.netD_1, self.real1, self.fake1) * self.opt.lambda_D1
        self.loss_D_2 = self.backward_D_basic(self.netD_2, self.real2, self.fake2) * self.opt.lambda_D2
        self.loss_D_3 = self.backward_D_basic(self.netD_3, self.real3, self.fake3) * self.opt.lambda_D3

        loss_sum = self.loss_D_1 + self.loss_D_2 + self.loss_D_3
        return loss_sum

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_As"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_par = self.backward_D_par() * self.opt.lambda_part_adv
        self.loss_D_A_all = self.loss_D_A + self.loss_D_par
        self.loss_D_A_all.backward()


    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B.backward()


    def backward_G(self):
        """Calculate the loss for generators G"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_par = self.opt.lambda_par
        lambda_gra = self.opt.lambda_gra
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B,self.graft_real_B,self.majormask_realB,self.ft_list_real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0


        lam_par = self.opt.lambda_part_adv
        #extra adversarial loss for G_ZY :
        self.loss_G_par = self.criterionGAN(self.netD_1(self.fake1), True) * self.opt.lambda_D1 *lam_par + \
                          self.criterionGAN(self.netD_2(self.fake2), True) * self.opt.lambda_D2 * lam_par + \
                          self.criterionGAN(self.netD_3(self.fake3), True) * self.opt.lambda_D3 * lam_par

        #GAN loss (adversarial loss)
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        #cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # Unsupervised semantic loss. Eq.7
        self.loss_par_A = self.criterionL1(self.par_B,self.par_A_fake) * lambda_par
        self.loss_par_B = self.criterionL1(self.par_A,self.par_B_fake) * lambda_par

        #statistical gradient loss
        self.loss_gra = self.cal_his(self.graavg_real_B[0],self.graavg_syn[0]) + \
                        self.cal_his(self.graavg_real_B[1], self.graavg_syn[1]) + \
                        self.cal_his(self.graavg_real_B[2],self.graavg_syn[2])
        self.loss_gra = self.loss_gra * lambda_gra /3.0

        #pixel-wise gradient loss
        self.loss_gra_L1 = self.criterionL1(self.graimg_real_A,self.grasyn_real_A) *self.opt.lambda_gra_L1
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B + self.loss_gra +self.loss_gra_L1 +  \
                      self.loss_par_A + self.loss_par_B + self.loss_G_par
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_1, self.netD_2, self.netD_3], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set Gs' gradients to zero
        self.backward_G()             # calculate gradients for Gs
        self.optimizer_G.step()       # update Gs' weights
        # D
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_1, self.netD_2, self.netD_3], True)
        self.optimizer_D.zero_grad()   # set Ds' gradients to zero
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()  # update Ds weights



class Gradient_Net(nn.Module):
    def __init__(self):
        super(Gradient_Net, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).expand(3,1,3,3)

        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).expand(3, 1, 3, 3)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x,padding=1, groups=3)
        grad_y = F.conv2d(x, self.weight_y,padding=1, groups=3)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient
