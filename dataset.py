import os
import re
import nrrd
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandZoomd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    ToTensord,
    ScaleIntensityRanged
)
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class MyNRRDDataSet(Dataset):
    """自定义NRRD格式数据集，直接加载和处理为 (64,64,64) 形状的图像"""

    def __init__(self, root_dir: str, split: str, radiomic_csv: str, target_shape=(64,64,64), num_augmentations=8):
        """
        Args:
            root_dir (str): 数据集的根目录
            split (str): 数据集划分，'train' 或 'test'
            radiomic_csv (str): 包含放射组学特征的CSV文件路径
            transform (callable, optional): 可选的样本转换
            target_shape (tuple, optional): 目标形状 (D, H, W)，默认调整到 (64,64,64)
            num_augmentations (int): 每个原始图像生成的增强样本数量
        """
        data_transform = Compose([
            RandFlipd(keys=["image"], spatial_axis=[1], prob=0.5),  # 随机X轴翻转
            RandFlipd(keys=["image"], spatial_axis=[0], prob=0.5),  # 随机Y轴翻转
            RandRotate90d(keys=["image"], prob=0.5, max_k=3, spatial_axes=(0, 1)),  # 随机90度旋转
            RandZoomd(keys=["image"], prob=0.5, min_zoom=0.9, max_zoom=1.1),  # 随机缩放
            RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.01),  # 添加随机高斯噪声
            RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), sigma_z=(0.5, 1.15)),  # 高斯平滑
            RandShiftIntensityd(keys=["image"], prob=0.5, offsets=0.10),  # 随机强度平移
            ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),  # 强度缩放
            ToTensord(keys=["image"])  # 将数据转换为Tensor
        ])
        self.data_list = []  # 存储所有样本的图像数据和标签
        self.transform = data_transform
        self.target_shape = target_shape
        self.num_augmentations = num_augmentations  # 每个原始图像生成的增强样本数量

        # 加载并标准化放射组学特征
        radiomic_df = pd.read_csv(radiomic_csv, dtype={'ID': str}).set_index('ID')
        scaler = StandardScaler()
        self.radiomic_features = pd.DataFrame(scaler.fit_transform(radiomic_df), index=radiomic_df.index, columns=radiomic_df.columns)

        # 加载 0 和 1 文件夹中的数据
        self._load_images_from_folder(os.path.join(root_dir, split, '0'), label=0)  # NoMetastasis
        self._load_images_from_folder(os.path.join(root_dir, split, '1'), label=1)  # Metastasis

    def _load_images_from_folder(self, folder: str, label: int):
        """加载指定文件夹中的所有 NRRD 文件，并分配类别标签"""
        for filename in os.listdir(folder):
            if filename.endswith(".nrrd"):  # 假设所有文件都为 NRRD 图像文件
                img_path = os.path.join(folder, filename)
                print(f"Processing data for: {img_path}")
                img = self._process_nrrd(img_path)
                
                # 提取文件名前缀（sm1_或sm2_）和数字 ID 以匹配 CSV 中的放射组学特征
                match = re.match(r'(sm\d+)_(\d+)', filename)  # 匹配前缀和数字部分
                if match:
                    prefix = match.group(1)       # 提取前缀，例如 'sm1' 或 'sm2'
                    id_number = match.group(2)    # 提取数字 ID，例如 '78'
                    id_ = f"{prefix}_{id_number}_image"  # 生成完整的ID，例如 'sm1_78_image'

                    if id_ in self.radiomic_features.index:
                        radiomic_feature = self._get_radiomic_feature(id_)
                        self.data_list.append((img, label, radiomic_feature))  # 存储图像、标签及其放射组学特征
                    else:
                        print(f"Warning: Radiomic features for ID {id_} not found in CSV file.")
                else:
                    print(f"Warning: Filename {filename} does not match expected pattern 'sm<1 or 2>_<number>_image.nrrd'")


    def _process_nrrd(self, file_path):
        """处理 NRRD 文件并返回调整后的图像"""
        data, header = nrrd.read(file_path)

        # 转换为 PyTorch 张量，并确保数据格式为 (D, H, W)
        print(data.shape)
        img = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)  # 将 (H, W, D) 转换为 (D, H, W)

        # 确保输入是 3D 数据 (D, H, W)
        if img.ndim != 3:
            raise ValueError(f"Image at {file_path} is not a 3D volume.")

        # 线性插值到目标形状 (64,64,64)
        img = self.interpolate_to_shape(img, self.target_shape)
        
        return img  # 直接返回插值后的完整图像

    def _get_radiomic_feature(self, id_):
        """获取对应 ID 的放射组学特征，并标准化"""
        # 获取特定 ID 的特征
        features = self.radiomic_features.loc[id_].values
        
        # 计算每列的均值和标准差（假设已经预计算，可以直接使用）
        mean = self.radiomic_features.mean().values  # 每列均值
        std = self.radiomic_features.std().values  # 每列标准差
        
        # 标准化：Z = (X - mean) / std
        standardized_features = (features - mean) / (std + 1e-8)  # 避免除以 0
        
        # 返回标准化后的特征作为 PyTorch 张量
        return torch.tensor(standardized_features, dtype=torch.float32)


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img, label, radiomic_feature = self.data_list[idx]

        # 生成增强图像样本，并包含原始图像
        augmented_samples = []
        # 添加原始图像
        original_img = self.normalize(img).unsqueeze(0)  # 添加通道维度 (C)
        augmented_samples.append((original_img, label, radiomic_feature))

        # 生成增强样本
        if self.transform:
            for _ in range(self.num_augmentations):
                augmented_img = self.transform({"image": img})["image"]  # 确保输入为字典，输出取回图像
                augmented_img = self.normalize(augmented_img)
                augmented_img = augmented_img.unsqueeze(0)  # 添加通道维度 (C)
                augmented_samples.append((augmented_img, label, radiomic_feature))

        # 返回包含原始图像和多个增强样本的列表
        return augmented_samples

    def normalize(self, img, window_min=None, window_max=None):
        """
        将图像裁剪到指定窗口范围并归一化到 [0, 1] 范围。
        
        Args:
            img: 输入的图像张量 (D, H, W)
            window_min: 窗口下限（HU 值的最小值），默认为 None（使用图像最小值）。
            window_max: 窗口上限（HU 值的最大值），默认为 None（使用图像最大值）。
        
        Returns:
            归一化到 [0, 1] 的张量。
        """
        # 如果未指定窗口范围，使用图像的最小值和最大值
        if window_min is None:
            window_min = img.min().item()
        if window_max is None:
            window_max = img.max().item()

        # 裁剪图像到窗口范围
        img = torch.clamp(img, min=window_min, max=window_max)

        # 避免除以零的情况
        if window_max > window_min:
            img = (img - window_min) / (window_max - window_min)  # 归一化到 [0, 1]
        else:
            img = torch.zeros_like(img)  # 如果 max == min，直接返回全零张量

        return img


    def interpolate_to_shape(self, img, target_shape):
        """
        对输入的 3D 图像进行调整到指定形状
        Args:
            img: 输入图像张量 (D, H, W)
            target_shape: 目标形状 (target_D, target_H, target_W)
        Returns:
            调整后的张量
        """
        img = img.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
        img = F.interpolate(img, size=target_shape, mode='trilinear', align_corners=True)  # 线性插值到目标形状
        img = img.squeeze(0).squeeze(0)  # 移除批次和通道维度
        return img

    @staticmethod
    def collate_fn(batch):
        # 将所有样本展开为单一列表，包含原始和增强样本
        all_samples = [sample for sublist in batch for sample in sublist if len(sublist) > 0]
        if len(all_samples) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])  # 返回空张量以避免错误
        all_imgs, all_labels, all_features = zip(*all_samples)  # 解包图像、标签和放射组学特征
        all_imgs = torch.stack(all_imgs, dim=0)  # 堆叠所有图像
        all_labels = torch.as_tensor(all_labels)  # 转换标签为张量
        all_features = torch.stack(all_features, dim=0)  # 堆叠所有放射组学特征
        return all_imgs, all_labels, all_features  # 返回图像、标签和放射组学特征


if __name__ == "__main__":
    root_dir = r'/home/yuwenjing/data/sm_3'
    radiomic_csv = r'/home/yuwenjing/data/sm_3/selected_features.csv'

    # 创建数据集实例，调整图像大小为 (128, 128, 128)
    dataset = MyNRRDDataSet(root_dir, split='train', radiomic_csv=radiomic_csv, num_augmentations=8)

    # 使用 PyTorch DataLoader
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=MyNRRDDataSet.collate_fn)  # 设置 batch_size=1 查看单个样本的增强效果

    # 检查数据集样本总数
    print(f"数据集样本总数: {len(dataset)}")

    # 取一个样本并打印其信息
    for imgs, labels, features in dataloader:
        print("单个批次的信息：")
        print(f"批次中的图像数量（含增强样本）: {imgs.shape[0]}")
        print(f"每个图像的形状: {imgs[0].shape}")  # 打印单个图像的形状
        print(f"标签: {labels}")
        print(f"放射组学特征的形状: {features.shape}")

        # 检查是否加载了增强后的数据
        for i, img in enumerate(imgs):
            print(f"样本 {i + 1} 的图像最大值: {img.max()}, 最小值: {img.min()}")  # 可进一步验证是否包含增强的多样性
        break  # 仅查看一个批次，循环后立即退出