from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

nc=3 # 表明生成的图像将有3个颜色通道（RGB）。

#checking the availability of cuda devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# custom weights initialization called on netG and netD
def weights_init(m):
    """
    weights_init 函数用于自定义模型的权重初始化，用于初始化生成器和判别器的权重。
    对于卷积层（Conv包含在类名中），权重从均值为0、方差为0.02的正态分布中随机初始化。
    对于批量归一化层（BatchNorm包含在类名中），权重从均值为1、方差为0.02的正态分布中随机初始化，偏置项被设置为0。
    这样的初始化有助于模型的稳定训练。
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class DCGAN(nn.Module):
    """
    DCGAN类继承自nn.Module，定义了生成器网络的结构。
    """
    def __init__(self, ngpu):
        super(DCGAN, self).__init__()
        self.ngpu = ngpu # 可以用于训练的GPU数量
        self.nz = 100 # 噪声向量的维度，即生成器的输入大小。
        self.nc = 3 # 生成图像的通道数（本例中为3，对应RGB）。
        self.ngf = 64 # 生成器内部层的特征图数量的基础大小，后续层将此数乘以特定因子。
        """
        生成器的架构由一系列的nn.ConvTranspose2d卷积转置层（有时也称之为反卷积）组成
        这些层逐步将输入的噪声向量上采样至目标图像的尺寸
        每个卷积转置层之后除了最后一个外，都跟随着一个批量归一化层（nn.BatchNorm2d）和ReLU激活函数
        最后一个卷积转置层后使用Tanh激活函数将图像数据规范化到[-1, 1]区间。
        """
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        # 定义了将输入张量（噪声）通过网络转换为输出图像的过程
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
    
    def input2output(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        # 还将输出图像通过简单的变换规范化到[0, 1]区间，以便于图像的显示和保存。
        # 这种处理对于查看生成的图像尤其有用。
        output = (output / 2 + 0.5).clamp(0, 1)

        return output

