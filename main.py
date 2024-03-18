import torch
from torch.cuda.amp import GradScaler
from inference_utils import *
from inference_models import get_init_noise, get_model,from_noise_to_image
from inference_image0 import get_image0
import argparse
import numpy as np
import complexity
import cv2

# 首先，通过命令行参数解析器 argparse 定义了一些参数，如输入选择、距离度量、模型类型、学习率等。
parser = argparse.ArgumentParser()
parser.add_argument("--input_selection", default="", type=str, help="The path of dev set.")
parser.add_argument("--distance_metric", default="l2", type=str, help="The path of dev set.")
parser.add_argument("--model_type", default="ddpm_cifar10", type=str, help="The path of dev set.")
parser.add_argument("--model_path_", default=None, type=str, help="The path of dev set.")

parser.add_argument("--lr", default=1e-2, type=float, help="")
parser.add_argument("--dataset_index", default=None, type=int, help="")
parser.add_argument("--bs", default=8, type=int, help="")
parser.add_argument("--write_txt_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--num_iter", default=2000, type=int, help="The path of dev set.")
parser.add_argument("--strategy", default="mean", type=str, help="The path of dev set.")
parser.add_argument("--mixed_precision", action="store_true", help="The path of dev set.")
parser.add_argument("--sd_prompt", default=None, type=str, help="The path of dev set.")
parser.add_argument("--input_selection_url", default=None, type=str, help="The path of dev set.")
parser.add_argument("--input_selection_name", default=None, type=str, help="The path of dev set.")
parser.add_argument("--input_selection_model_type", default=None, type=str, help="The path of dev set.")
parser.add_argument("--input_selection_model_path", default=None, type=str, help="The path of dev set.")

args = parser.parse_args()

"""
这段代码是一个图像重构的优化过程，通过优化噪声输入来重构目标图像。
总的来说，这段代码通过迭代优化噪声输入，使用指定的模型和损失函数，逐步重构出与目标图像相似的图像。在优化过程中，记录最低损失值，并根据需要保存中间结果和最终结果。
"""

# 加载和初始化模型：通过 get_model 函数获取指定类型的模型，并将其赋值给 args.cur_model。
args.cur_model = get_model(args.model_type,args.model_path_,args)
# 加载初始图像和噪声数据：通过 get_image0 函数获取初始图像 image0 和真实噪声 gt_noise。
image0, gt_noise = get_image0(args)
image0 = image0.detach()
# 通过 get_init_noise 函数获取初始噪声 init_noise。
init_noise = get_init_noise(args,args.model_type,args.cur_model,bs=args.bs)

# 根据模型类型，选择适当的优化器和参数：对于不同的模型类型（如 "sd"、"sd_unet" 等），创建相应的噪声参数，并使用 Adam 优化器进行优化。
if args.model_type in ["sd"]:
    cur_noise = torch.nn.Parameter(torch.tensor(init_noise)).cuda()
    optimizer = torch.optim.Adam([cur_noise], lr=args.lr)
elif args.model_type in ["sd_unet"]:
    args.cur_model.unet.eval()
    args.cur_model.vae.eval()
    cur_noise_0 = torch.nn.Parameter(torch.tensor(init_noise[0])).cuda()
    optimizer = torch.optim.Adam([cur_noise_0], lr=args.lr)
else:
    cur_noise = torch.nn.Parameter(torch.tensor(init_noise)).cuda()
    optimizer = torch.optim.Adam([cur_noise], lr=args.lr)
    
# 根据指定的距离度量（如 L1、L2、SSIM 等），选择相应的损失函数 criterion。
if args.distance_metric == "l1":
    criterion = torch.nn.L1Loss(reduction='none')
elif args.distance_metric == "l2":
    criterion = torch.nn.MSELoss(reduction='none')
elif args.distance_metric == "ssim":
    criterion = SSIMLoss().cuda()
elif args.distance_metric == "psnr":
    criterion = psnr
elif args.distance_metric == "lpips":
    criterion = lpips_fn
    
import time
args.measure = float("inf")

# 如果启用了混合精度，使用 GradScaler 来管理梯度的缩放，以优化训练过程。
if args.mixed_precision:
    scaler = GradScaler()
    
experiment_folder_path = create_experiment_folder(args.input_selection)
    
# 在每个迭代步骤中：
for i in range(args.num_iter):
    start_time = time.time()
    print("step:",i)

    # 使用 from_noise_to_image 函数将噪声转换为图像。
    # 计算重构图像和目标图像之间的损失。
    if args.mixed_precision:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            image = from_noise_to_image(args,args.cur_model,cur_noise,args.model_type)
            loss = criterion(image0,image).mean()
    else:
        image = from_noise_to_image(args,args.cur_model,cur_noise,args.model_type)
        loss = criterion(image0.detach(),image).mean()

    # 每隔 100 步保存当前重构的图像。
    if i%100==0:
        epoch_num_str=str(i)
        with torch.no_grad():
            file_name = f"image_cur_{args.input_selection}_{args.distance_metric}_{str(args.lr)}_bs{str(args.bs)}{epoch_num_str}.png"
            save_image_to_experiment_folder(image, experiment_folder_path, file_name)
        
            
    # 根据指定的策略（如 "min" 或 "mean"），更新最小损失值 args.measure。
    min_value = criterion(image0,image).mean(-1).mean(-1).mean(-1).min()
    mean_value = criterion(image0,image).mean()
    if (args.strategy == "min") and (min_value < args.measure):
        args.measure = min_value
    if (args.strategy == "mean") and (mean_value < args.measure):
        args.measure = mean_value
    
    # 打印当前的损失值。
    print("lowest loss now:",args.measure.item())
    if args.distance_metric == "lpips":
        loss = loss.mean()
    print("loss "+args.input_selection+" "+args.distance_metric+":",loss.item())
    
    write_loss_to_csv(experiment_folder_path, i, loss.item(), min_value.item(), mean_value.item())


    # 根据是否使用混合精度，使用相应的方式进行梯度计算和优化器更新。
    if args.mixed_precision:
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 打印一次迭代的时间。
    end_time = time.time()
    print("time for one iter: ",end_time-start_time)
    # 清空 CUDA 缓存。
    torch.cuda.empty_cache()

# 计算最终的最低损失值，并打印。
print("*"*80)
print("final lowest loss: ",args.measure.item())
# 将目标图像转换为灰度图像，并计算其 2D 熵作为复杂度度量。
cv2_img0 = (image0.squeeze(0).permute(1, 2, 0).cpu().numpy()* 255).astype(np.uint8)
cv2_img0 = cv2.cvtColor(cv2_img0, cv2.COLOR_BGR2GRAY)
print("2D-entropy-based complexity: ", complexity.calcEntropy2dSpeedUp(cv2_img0, 3, 3))

# 如果指定了写入文本文件的路径，将最低损失值写入文件。
if args.write_txt_path:
    with open(args.write_txt_path,"a") as f:
        f.write(str(args.measure.item())+"\n")

# 根据不同的输入选择（如 SD 提示或输入选择 URL），保存原始图像和最终重构的图像。
if args.sd_prompt:
    save_img_tensor(image0,"./result_imgs/ORI_"+args.sd_prompt+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")
    save_img_tensor(image,"./result_imgs/last_"+args.sd_prompt+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")
if args.input_selection_url:
    save_img_tensor(image0,"./result_imgs/ORI_"+args.input_selection_url.split("/")[-1]+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")
    save_img_tensor(image,"./result_imgs/last_"+args.input_selection_url.split("/")[-1]+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")
