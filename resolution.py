import argparse
from PIL import Image

def calculate_image_resolution(image_path):
    """
    计算并返回图像的分辨率。

    参数:
    image_path: 图像文件的路径。

    返回:
    (width, height): 一个元组，包含图像的宽度和高度。
    """
    # 使用Pillow库打开图像
    with Image.open(image_path) as img:
        # 获取图像的宽度和高度
        width, height = img.size

    return width, height

def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description='计算图像分辨率的工具')
    # 添加图像路径参数
    parser.add_argument('image_path', type=str, help='图像文件的路径')
    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数计算分辨率
    resolution = calculate_image_resolution(args.image_path)
    print(f"图像分辨率为: {resolution[0]}x{resolution[1]}")

if __name__ == "__main__":
    main()
