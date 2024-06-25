import os
import shutil

def move_images_to_root_directory(root_dir):
    # 遍历根目录下的所有子目录和文件
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            # 构建文件的完整路径
            file_path = os.path.join(subdir, file)
            # 检查文件是否是图片（这里假设图片扩展名为.jpg或.png，可以根据需要调整）
            if file.endswith('.jpg') or file.endswith('.png'):
                # 构建目标路径
                target_path = os.path.join(root_dir, file)
                # 移动文件
                shutil.move(file_path, target_path)
                print(f"Moved {file} to {target_path}")

# 指定根目录路径
root_directory = '/data0/sunyj/dataset/LSDIR/train/HR'
move_images_to_root_directory(root_directory)
