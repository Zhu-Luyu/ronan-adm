import argparse
import os
import numpy as np
import tensorflow as tf
import torch as th
import torch.optim as optim
from PIL import Image
import cv2
from datetime import datetime
import pickle

from my_utils import create_time_named_dir

from inference_utils import *

# TensorFlow设置
tf.compat.v1.InteractiveSession()

# 导入GAN模型
with open('karras2018iclr-lsun-bedroom-256x256.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

# 定义PyTorch设备
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# 参数解析器
parser = argparse.ArgumentParser()
parser.add_argument("--input_selection", default="", type=str, help="The path of dev set.")
parser.add_argument("--input_selection_name", type=str, default="path/to/your/input/image.jpg")
parser.add_argument("--num_iter", default=2000, type=int, help="The path of dev set.")
parser.add_argument("--lr", default=1e-2, type=float, help="")
args = parser.parse_args()

# 读取并处理样本图像
image0 = cv2.imread(args.input_selection_name)
image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
image0 = cv2.resize(image0, (256, 256)) / 255.0
image0 = th.tensor(image0, dtype=th.float32).permute(2, 0, 1).unsqueeze(0).to(device)

# 初始化噪声向量
latent = th.randn((1, *Gs.input_shapes[0][1:]), device=device, requires_grad=True)
optimizer = optim.Adam([latent], lr=args.lr)

criterion = torch.nn.MSELoss(reduction='none')

# save_dir = create_time_named_dir()
experiment_folder_path = create_experiment_folder(args.input_selection)

# 追踪最低损失
lowest_loss = None
best_img = None

# 开始优化循环
for i in range(args.num_iter):
    optimizer.zero_grad()
    
    # 用噪声生成图像，注意这里将latent先detach再转换为numpy数组
    generated_img = Gs.run(latent.detach().cpu().numpy(), np.zeros([1] + Gs.input_shapes[1][1:]))
    generated_img = np.clip(np.rint((generated_img + 1.0) / 2.0 * 255.0), 0, 255).astype(np.uint8)
    # generated_img_tensor = th.tensor(generated_img, dtype=th.float32).permute(0, 3, 1, 2).to(device) / 255.0
    generated_img_tensor = th.tensor(generated_img, dtype=th.float32).to(device) / 255.0

    print(f"generated_img_tensor shape: {generated_img_tensor.shape}")
    print(f"image0 shape: {image0.shape}")

    # 计算损失并反向传播
    loss = criterion(generated_img_tensor, image0)
    loss.backward()
    optimizer.step()
    
    # 每隔一定步数保存生成的图像
    if i % 5 == 0 or i == args.num_iter - 1:
        print(f"Step: {i}, Loss: {loss.item()}")
        save_img = generated_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        save_img = (save_img * 255).astype(np.uint8)
        save_path = os.path.join(save_dir, f"generated_step_{i}.png")
        Image.fromarray(save_img).save(save_path)
        
    # 更新最低损失和保存最佳图像
    if lowest_loss is None or loss < lowest_loss:
        lowest_loss = loss
        best_img = generated_img_tensor
        
    
# 保存损失最小的图像
save_img = best_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
save_img = (save_img * 255).astype(np.uint8)
save_path = os.path.join(experiment_folder_path, "best_image.png")
Image.fromarray(save_img).save(save_path)
print(f"final lowest loss: {lowest_loss.item()}")

