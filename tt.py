# import os

# def count_nrrd_files(directory):
#     nrrd_count = 0
#     # Walk through all subdirectories and files in the given directory
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith('.nrrd'):
#                 nrrd_count += 1
#     return nrrd_count

# # 指定文件夹路径
# directory = '/home/yuwenjing/data/肾母细胞瘤CT数据_划分/train/Metastasis'

# # 统计并输出nrrd文件的数量
# nrrd_file_count = count_nrrd_files(directory)
# print(f"Number of .nrrd files in '{directory}': {nrrd_file_count}")


# import os
# import shutil

# def copy_and_rename_nrrd_files(source_dirs, dest_dir):
#     # 确保目标目录存在，如果不存在则创建
#     if not os.path.exists(dest_dir):
#         os.makedirs(dest_dir)

#     # 新的序号从1开始
#     file_counter = 1

#     for source_dir in source_dirs:
#         for root, dirs, files in os.walk(source_dir):
#             for file in files:
#                 # 只保留以 _image.nrrd 结尾的文件
#                 if file.endswith('_image.nrrd'):
#                     # 新的文件名是连续的数字
#                     new_file_name = f"{file_counter}_image.nrrd"
#                     source_file_path = os.path.join(root, file)
#                     dest_file_path = os.path.join(dest_dir, new_file_name)

#                     # 复制文件并重命名
#                     shutil.copy2(source_file_path, dest_file_path)
#                     print(f"Copied {file} to {dest_file_path}")

#                     # 增加计数器
#                     file_counter += 1

# # 指定源目录和目标目录
# source_dirs = [
#     '/home/yuwenjing/data/肾母细胞瘤CT数据/Data1',
#     '/home/yuwenjing/data/肾母细胞瘤CT数据/Data3-1'
# ]

# dest_dir = '/home/yuwenjing/data/肾母细胞瘤CT数据/data1'

# # 调用函数复制并重命名nrrd文件
# copy_and_rename_nrrd_files(source_dirs, dest_dir)

# print("All _image.nrrd files have been copied and renamed successfully.")


# import os
# import shutil
# import random

# def split_data(source_dir, dest_dir, label, train_ratio=0.8):
#     # 获取源目录下所有文件
#     files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
#     # 随机打乱文件列表
#     random.shuffle(files)
    
#     # 计算训练集和测试集的分割点
#     train_size = round(len(files) * train_ratio)
    
#     # 创建目标目录
#     train_dir = os.path.join(dest_dir, 'train', label)
#     test_dir = os.path.join(dest_dir, 'test', label)
#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(test_dir, exist_ok=True)
    
#     # 将前train_size个文件放入train文件夹
#     for file in files[:train_size]:
#         shutil.copy2(os.path.join(source_dir, file), os.path.join(train_dir, file))
#         print(f"Copied {file} to {train_dir}")
    
#     # 将剩下的文件放入test文件夹
#     for file in files[train_size:]:
#         shutil.copy2(os.path.join(source_dir, file), os.path.join(test_dir, file))
#         print(f"Copied {file} to {test_dir}")

# # 指定源目录和目标目录
# data0_dir = '/home/yuwenjing/data/肾母细胞瘤CT数据/data0'  # NoMetastasis
# data1_dir = '/home/yuwenjing/data/肾母细胞瘤CT数据/data1'  # Metastasis
# output_dir = '/home/yuwenjing/data/肾母细胞瘤CT数据_划分'

# # 设置随机种子确保结果可重复
# random.seed(42)

# # 调用函数分别处理data0和data1
# split_data(data0_dir, output_dir, 'NoMetastasis', train_ratio=0.8)
# split_data(data1_dir, output_dir, 'Metastasis', train_ratio=0.8)

# print("Data has been successfully split into train and test folders.")



# import os
# import nrrd

# def read_nrrd_depth(file_path):
#     # 读取nrrd文件
#     data, header = nrrd.read(file_path)
    
#     # 获取数据形状并打印深度维度
#     depth = data.shape[-1] if len(data.shape) == 3 else None
#     print(f"文件: {file_path}")
#     print(f"数据形状: {data.shape}")
#     print(f"深度维数: {depth}\n")
#     return depth

# def process_folder(folder_path):
#     # 遍历文件夹，查找所有 .nrrd 文件
#     nrrd_files = [f for f in os.listdir(folder_path) if f.endswith('.nrrd')]
    
#     if not nrrd_files:
#         print("该文件夹中没有找到 .nrrd 文件。")
#         return

#     # 统计所有nrrd文件的深度维度信息
#     for nrrd_file in nrrd_files:
#         file_path = os.path.join(folder_path, nrrd_file)
#         read_nrrd_depth(file_path)

# if __name__ == "__main__":
#     # 指定要读取的文件夹路径
#     folder_path = r'/home/yuwenjing/data/肾母细胞瘤CT数据_划分/train/Metastasis'
    
#     # 调用函数处理文件夹中的所有nrrd文件
#     process_folder(folder_path)

# import os
# import nrrd
# import torch
# from torch.utils.data import Dataset
# import numpy as np

# class MyNRRDDataSet(Dataset):
#     """自定义NRRD格式数据集，针对3D卷积适应 (D, H, W) 格式"""

#     def __init__(self, root_dir: str, split: str, transform=None):
#         """
#         Args:
#             root_dir (str): 数据集的根目录
#             split (str): 数据集划分，'train' 或 'test'
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.block_list = []  # 存储所有样本的体素数据和标签
#         self.transform = transform
#         self.background_value = None  # 用于存储背景体素值

#         # 加载无转移组 (NoMetastasis) 和 有转移组 (Metastasis) 的数据
#         self._load_images_from_folder(os.path.join(root_dir, split, 'NoMetastasis'), label=0)
#         self._load_images_from_folder(os.path.join(root_dir, split, 'Metastasis'), label=1)

#         # 在加载完所有数据后计算背景体素值
#         self.background_value = self.calculate_background_value()

#     def _load_images_from_folder(self, folder: str, label: int):
#         """加载指定文件夹中的所有 NRRD 文件，并分配类别标签"""
#         for filename in os.listdir(folder):
#             if filename.endswith(".nrrd"):  # 假设所有文件都为图像文件
#                 img_path = os.path.join(folder, filename)
#                 print(f"Processing data for: {img_path}")
#                 blocks = self._process_nrrd(img_path)  
#                 self.block_list.extend([(block.clone().detach(), label) for block in blocks]) 

#     def _process_nrrd(self, file_path):
#         """处理 NRRD 文件并返回整个体素数据"""
#         data, header = nrrd.read(file_path)

#         # 转换为 PyTorch 张量，并确保数据格式为 (D, H, W)
#         img = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)  # 将 (H, W, D) 转换为 (D, H, W)

#         # 确保输入是 3D 数据 (D, H, W)
#         if img.ndim != 3:
#             raise ValueError(f"Image at {file_path} is not a 3D volume.")

#         return [img]  # 直接返回图像块

#     def calculate_background_value(self):
#         """
#         计算所有加载数据的背景体素值，假设背景体素值是出现次数最多的体素值。
#         Returns:
#             background_value: 出现次数最多的体素值
#         """
#         all_voxels = torch.cat([block.flatten() for block, _ in self.block_list])
#         # 使用 numpy 计算直方图并返回出现次数最多的体素值
#         unique_vals, counts = torch.unique(all_voxels, return_counts=True)
#         background_value = unique_vals[torch.argmax(counts)].item()
#         print(f"Background voxel value: {background_value}")
#         return background_value

#     def __len__(self):
#         return len(self.block_list)

#     def __getitem__(self, idx):
#         block, label = self.block_list[idx]  

#         # 如果有 transform，应用 transform 到该块
#         if self.transform:
#             block = self.transform(block)

#         # 对图像块进行标准化
#         block = self.normalize(block)  # 在这里调用标准化函数

#         # 添加通道维度 (C)
#         block = block.unsqueeze(0)
#         return block, label  

#     def normalize(self, img):
#         """
#         将图像归一化到 [0, 1] 范围
#         Args:
#             img: 输入的图像张量 (D, H, W)
#         Returns:
#             归一化到 [0, 1] 的张量
#         """
#         min_val = img.min()
#         max_val = img.max()

#         # 避免除以零的情况
#         if max_val > min_val:
#             img = (img - min_val) / (max_val - min_val)
#         else:
#             img = torch.zeros_like(img)  # 如果 max == min，直接返回全零张量

#         return img

#     @staticmethod
#     def collate_fn(batch):
#         all_blocks, all_labels = zip(*batch)  # 只解包 block 和 label
#         all_blocks = torch.stack(all_blocks, dim=0)  # 堆叠所有分块图像
#         all_labels = torch.as_tensor(all_labels)     # 转换标签为张量
#         return all_blocks, all_labels  # 不再返回掩码


# # 使用示例
# if __name__ == "__main__":
#     root_dir = r'/home/yuwenjing/data/tmp_data'

#     # 创建数据集实例
#     dataset = MyNRRDDataSet(root_dir, split='test')

#     # 检查数据集总数（块数）
#     print(f"数据集样本总数（块数）: {len(dataset)}")

#     # 打印背景体素值
#     print(f"背景体素值: {dataset.background_value}")

import nrrd
import os
import numpy as np
from PIL import Image
import glob

def save_nrrd_slices_as_images(nrrd_file, output_dir, image_format="png"):
    """
    将一个 NRRD 文件的所有切片保存为图像文件
    
    Args:
        nrrd_file (str): NRRD 文件的路径
        output_dir (str): 输出图片的保存文件夹
        image_format (str): 输出图像的格式（默认 png）
    """
    # 读取 NRRD 文件
    data, header = nrrd.read(nrrd_file)

    # 检查并创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历每个切片，并将其保存为图片
    depth = data.shape[2]  # 获取切片数量
    for i in range(depth):
        slice_2d = data[:, :, i]  # 提取单个切片 (H, W)

        # 归一化切片以适应 0-255 的像素值范围
        slice_2d_normalized = ((slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d)) * 255).astype(np.uint8)

        # 将切片转换为图像对象
        img = Image.fromarray(slice_2d_normalized)

        # 保存图片，文件名为 slice_i.png（或其他格式）
        img.save(os.path.join(output_dir, f"slice_{i}.{image_format}"))

    print(f"所有切片已保存至 {output_dir}")

def delete_png_files(folder_path):
    """
    删除指定文件夹下的所有 .png 文件

    Args:
        folder_path (str): 要删除 .png 文件的文件夹路径
    """
    # 获取所有 .png 文件的路径
    png_files = glob.glob(os.path.join(folder_path, "*.png"))

    # 遍历并删除每个 .png 文件
    for file_path in png_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

# 使用示例
folder_path = "/home/yuwenjing/DeepLearning/VIT_3d"  # 替换为实际的文件夹路径
# delete_png_files(folder_path)
# # 使用示例
nrrd_file = "/home/yuwenjing/data/肾母细胞瘤CT数据/Data3-1/1_label.nrrd"  # 替换为你的 NRRD 文件路径
output_dir = "/home/yuwenjing/DeepLearning/VIT_3d"  # 替换为你的输出文件夹
# save_nrrd_slices_as_images(nrrd_file, output_dir)

import os
import nrrd
import numpy as np
from PIL import Image

def save_non_black_image_and_label_slices(nrrd_image_file, nrrd_label_file, output_dir, image_format="png"):
    """
    若 image 和 label 的切片数一样，则保留 label 切片非全黑对应序号的 image 和 label 切片，分别保存为图片
    
    Args:
        nrrd_image_file (str): image 文件的路径
        nrrd_label_file (str): label 文件的路径
        output_dir (str): 输出图片的保存文件夹
        image_format (str): 输出图像的格式（默认 png）
    """
    # 读取 image 和 label 的 nrrd 文件
    image_data, image_header = nrrd.read(nrrd_image_file)
    label_data, label_header = nrrd.read(nrrd_label_file)

    # 检查 image 和 label 的切片数是否一致
    if image_data.shape[2] != label_data.shape[2]:
        print(f"Error: image 和 label 的切片数量不一致: {nrrd_image_file} 和 {nrrd_label_file}")
        return
    
    # 检查并创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历每个切片
    depth = image_data.shape[2]  # 切片数
    for i in range(depth):
        label_slice = label_data[:, :, i]  # 提取 label 的单个切片 (H, W)
        
        # 判断该 label 切片是否非全黑（即至少有一个像素值不是 0）
        if np.any(label_slice != 0):
            # 提取对应的 image 切片
            image_slice = image_data[:, :, i]

            # 归一化 image 切片以适应 0-255 的像素值范围
            image_slice_normalized = ((image_slice - np.min(image_slice)) / (np.max(image_slice) - np.min(image_slice)) * 255).astype(np.uint8)

            # 将 image 切片转换为图像对象
            img = Image.fromarray(image_slice_normalized)

            # 保存 image 切片，文件名为 image_slice_i.png（或其他格式）
            img.save(os.path.join(output_dir, f"image_slice_{i}.{image_format}"))

            # 归一化 label 切片到 0-255（方便可视化），label 切片值通常是分类标签，因此这里不严格归一化
            label_slice_normalized = (label_slice * 255).astype(np.uint8)

            # 将 label 切片转换为图像对象
            label_img = Image.fromarray(label_slice_normalized)

            # 保存 label 切片，文件名为 label_slice_i.png（或其他格式）
            label_img.save(os.path.join(output_dir, f"label_slice_{i}.{image_format}"))

    print(f"保存的 image 和 label 图片已存储在: {output_dir}")

# 使用示例
nrrd_image_file = "/home/yuwenjing/data/肾母细胞瘤CT数据/Data3-1/1_image.nrrd"  # 替换为 image 的 nrrd 文件路径
nrrd_label_file = "/home/yuwenjing/data/肾母细胞瘤CT数据/Data3-1/1_label.nrrd"  # 替换为 label 的 nrrd 文件路径
output_dir = "/home/yuwenjing/DeepLearning/VIT_3d"  # 替换为你的输出文件夹
# save_non_black_image_and_label_slices(nrrd_image_file, nrrd_label_file, output_dir)

