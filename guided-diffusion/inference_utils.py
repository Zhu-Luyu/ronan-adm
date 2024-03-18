"""
copy from ../inference_utils.py
"""
import torch

import torchvision
from torchvision import transforms

from piqa import SSIM
import lpips

import os
import time

import csv

"""
这段代码定义了几个函数和类，用于计算图像之间的相似性和质量度量。
引入一些外部库和模块，如 torch、torchvision、piqa、lpips 等，用于图像处理和相似性度量的计算。
总的来说，这段代码提供了几个有用的函数和类，用于评估和比较图像之间的相似性和质量。
*SSIMLoss 类用于计算 SSIM 损失，
*psnr 函数用于计算 PSNR 值，
*lpips_fn 函数用于计算 LPIPS 值。
这些度量可以用于评估生成的图像与真实图像之间的相似性，或者比较不同算法生成的图像的质量。
save_img_tensor 函数提供了一个方便的方式将 PyTorch 张量格式的图像保存到文件中。
"""

class SSIMLoss(SSIM):
    """
    继承自 SSIM 类，用于计算结构相似性指数（Structural Similarity Index Measure，SSIM）损失。
    重写了 forward 方法，将原始的 SSIM 值转换为损失值。
    当两幅图像越相似（SSIM 值越高），损失值越小。
    """
    def forward(self, x, y):
        return 1. - super().forward(x, y) # 当两幅图像越相似（SSIM 值越高），损失值越小
    
def psnr(img1, img2):
    """
    用于计算峰值信噪比（Peak Signal-to-Noise Ratio，PSNR）。
    将输入的两幅图像 img1 和 img2 乘以 255，将像素值范围从 [0, 1] 转换为 [0, 255]。
    计算两幅图像之间的均方误差（Mean Squared Error，MSE）。
    使用公式 20 * log10(255 / sqrt(MSE)) 计算 PSNR 值。
    PSNR 值越高，表示图像质量越好。
    """
    img1 = img1*255
    img2 = img2*255
    #mse = torch.mean((img1 - img2) ** 2)
    mse = ((img1 - img2)**2).mean(-1).mean(-1).mean(-1)
    return 20 * torch.log10(255.0 / torch.sqrt(mse)).mean() # PSNR 值越高，表示图像质量越好

loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
def lpips_fn(img1, img2):
    """
    使用 LPIPS（Learned Perceptual Image Patch Similarity）度量来计算两幅图像之间的感知相似性。
    首先将输入的两幅图像 img1 和 img2 的像素值范围从 [0, 1] 转换为 [-1, 1]。
    使用预训练的 VGG 网络作为特征提取器，计算两幅图像在特征空间上的距离。
    返回计算得到的 LPIPS 值，值越小表示图像在感知上越相似。
    """
    img1 = (img1 - 0.5)*2
    img2 = (img2 - 0.5)*2
    return loss_fn_vgg(img1,img2)

def save_img_tensor(img,name):
    """
    用于将 PyTorch 张量格式的图像保存到文件。
    使用 torchvision.utils.save_image 函数将图像张量保存为文件。
    保存的图像文件名由 name 参数指定。
    """
    #img = (img / 2 + 0.5).clamp(0, 1)
    torchvision.utils.save_image(img, name)
    #img = img.cpu().permute(0, 2, 3, 1).numpy()
    #img = ddim.numpy_to_pil(img)[0]
    #img.save(name)
    
def create_experiment_folder(input_selection):
    """
    创建一个新的文件夹，用于保存实验结果。文件夹名称包含实验的关键信息和开始时间。
    """
    experiment_start_time = time.strftime("%Y%m%d-%H%M%S")
    folder_name = f"experiment_{input_selection}_{experiment_start_time}"
    folder_path = os.path.join("./result_imgs", folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def save_image_to_experiment_folder(image, folder_path, file_name):
    """
    在指定的文件夹中保存图像。
    """
    save_path = os.path.join(folder_path, file_name)
    save_img_tensor(image, save_path)
    
def save_loss_to_experiment_folder(step, loss, folder_path, file_name):
    with open(f"{folder_path}/{file_name}", "a") as f:
        f.write(f"{step},{loss}\n")
        
def write_loss_to_csv(experiment_folder_path, step, loss, min_loss, mean_loss):
    csv_file_path = os.path.join(experiment_folder_path, 'losses.csv')
    file_exists = os.path.isfile(csv_file_path)
    
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = ['step', 'loss', 'min_loss', 'mean_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({'step': step, 'loss': loss, 'min_loss': min_loss, 'mean_loss': mean_loss})