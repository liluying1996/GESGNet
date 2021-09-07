# Unsupervised Face Super-Resolution via Gradient Enhancement and Semantic Guidance (GESGNet)
# Preparation

1. prepare unpaired data, which includes images of HR domain and LR domain
2. use [ESRGAN](https://github.com/xinntao/BasicSR) to transform LR domain into inter-HR domain
3. put images of inter-HR and HR domain in a folder `your_datasets`, and rename them as `trainA` and `trainB`, respectively.
4. download [the pretrained face parsing model](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view), and put it in the folder `models`. (The original face parsing codes can be found in https://github.com/zllrunning/face-parsing.PyTorch)
# Training
`python train.py --name experiment_name --dataroot /path/to/your_datasets`
# Testing
`python test.py --name experiment_name --dataroot /path/to/your_datasets`
