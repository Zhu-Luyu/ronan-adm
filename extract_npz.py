import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import csv  # 导入csv模块

def extract_and_save_images_and_labels(npz_path):
    # 加载npz文件
    data = np.load(npz_path)
    images = data['images']
    labels = data['labels']

    # 创建保存图像和标签的目录
    save_dir = os.path.splitext(npz_path)[0]
    os.makedirs(save_dir, exist_ok=True)

    # 准备保存所有标签的CSV文件
    labels_path = os.path.join(save_dir, 'labels.csv')
    with open(labels_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Image Index", "Label"])  # 写入表头
        # 遍历所有图像和标签，保存它们
        for idx in range(images.shape[0]):
            # 保存图像
            image_path = os.path.join(save_dir, f'image_{idx}.png')
            plt.imshow(images[idx])
            plt.axis('off')
            plt.savefig(image_path)
            plt.close()

            # 将标签写入CSV文件
            writer.writerow([idx, labels[idx]])

    print(f"All images have been saved to {save_dir}")
    print(f"All labels have been saved to {labels_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save images and labels from a .npz file.")
    parser.add_argument("npz_path", type=str, help="Path to the .npz file containing images and labels.")

    args = parser.parse_args()

    extract_and_save_images_and_labels(args.npz_path)
