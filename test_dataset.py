import os
import nrrd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MyNRRDDataSet(Dataset):
    """自定义NRRD格式数据集，直接加载和处理为 (128, 128, 128) 形状的图像"""

    def __init__(self, root_dir: str, split: str, transform=None, target_shape=(128, 128, 128), num_augmentations=1):
        """
        Args:
            root_dir (str): 数据集的根目录
            split (str): 数据集划分，'train' 或 'test'
            transform (callable, optional): 可选的样本转换
            target_shape (tuple, optional): 目标形状 (D, H, W)，默认调整到 (128, 128, 128)
            num_augmentations (int): 每个原始图像生成的增强样本数量
        """
        self.data_list = []  # 存储所有样本的图像数据和标签
        self.transform = transform
        self.target_shape = target_shape
        self.num_augmentations = num_augmentations  # 每个原始图像生成的增强样本数量

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
                self.data_list.append((img, label))  # 直接存储图像及其标签

    def _process_nrrd(self, file_path):
        """处理 NRRD 文件并返回调整后的图像"""
        data, header = nrrd.read(file_path)

        # 转换为 PyTorch 张量，并确保数据格式为 (D, H, W)
        print(data.shape)
        img = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)  # 将 (H, W, D) 转换为 (D, H, W)

        # 确保输入是 3D 数据 (D, H, W)
        if img.ndim != 3:
            raise ValueError(f"Image at {file_path} is not a 3D volume.")

        # 线性插值到目标形状 (128, 128, 128)
        img = self.interpolate_to_shape(img, self.target_shape)
        
        return img  # 直接返回插值后的完整图像

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img, label = self.data_list[idx]

        # 生成多个增强图像样本
        augmented_imgs = []
        if self.transform:
            for _ in range(self.num_augmentations):
                augmented_img = self.transform({"image": img})["image"]  # 确保输入为字典，输出取回图像
                augmented_img = self.normalize(augmented_img)
                augmented_img = augmented_img.unsqueeze(0)  # 添加通道维度 (C)
                augmented_imgs.append((augmented_img, label))

        # 返回包含多个增强样本的列表
        return augmented_imgs

    def normalize(self, img):
        """
        将图像归一化到 [0, 1] 范围
        Args:
            img: 输入的图像张量 (D, H, W)
        Returns:
            归一化到 [0, 1] 的张量
        """
        min_val = img.min()
        max_val = img.max()
        # 避免除以零的情况
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
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
        all_samples = [sample for sublist in batch for sample in sublist]
        all_imgs, all_labels = zip(*all_samples)  # 只解包图像和标签
        all_imgs = torch.stack(all_imgs, dim=0)  # 堆叠所有图像
        all_labels = torch.as_tensor(all_labels)  # 转换标签为张量
        return all_imgs, all_labels  # 返回图像和标签

# 使用示例
if __name__ == "__main__":
    root_dir = r'/home/yuwenjing/data/tmp_data'

    # 创建数据集实例，直接调整到 (128, 128, 128)
    dataset = MyNRRDDataSet(root_dir, split='test', num_augmentations=3)

    # 使用 PyTorch DataLoader
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=MyNRRDDataSet.collate_fn)

    # 检查数据集样本总数
    print(f"数据集样本总数: {len(dataset)}")

    # 打印批次数据的形状
    for imgs, labels in dataloader:
        print(f"批次图像的形状: {imgs.shape}")
        print(f"批次标签: {labels}")
