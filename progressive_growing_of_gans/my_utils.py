# my_utils.py

import os
from datetime import datetime

def create_time_named_dir(base_dir='progan_imgs'):
    """
    创建一个以当前时间命名的目录。

    :param base_dir: 基础目录的名称，默认为'progan_imgs'。
    :return: 新创建的时间命名目录的路径。
    """
    # 获取当前时间，格式化为字符串
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # 构建目标目录路径
    dir_path = os.path.join(base_dir, current_time)
    # 创建目标目录
    os.makedirs(dir_path, exist_ok=True)
    # 返回新创建的目录路径
    return dir_path