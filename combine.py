import os
import shutil
import pandas as pd

def create_folder_structure(base_dir):
    """
    在目标路径下创建 train 和 test 文件夹结构。
    """
    for split in ['train', 'test']:
        for label in ['0', '1']:
            os.makedirs(os.path.join(base_dir, split, label), exist_ok=True)

def copy_and_rename_files(src_base_dir, dst_base_dir, prefix):
    """
    复制并重命名源文件夹中的 nrrd 文件到目标文件夹，并添加前缀。
    """
    for split in ['train', 'test']:
        for label in ['0', '1']:
            src_dir = os.path.join(src_base_dir, split, label)
            dst_dir = os.path.join(dst_base_dir, split, label)
            
            # 遍历源文件夹中的每个文件
            for filename in os.listdir(src_dir):
                if filename.endswith(".nrrd"):
                    # 给文件添加前缀避免冲突，并复制到目标文件夹
                    new_filename = f"{prefix}_{filename}"
                    shutil.copy(os.path.join(src_dir, filename), os.path.join(dst_dir, new_filename))

def merge_radiomic_features(src_base_dir_1, src_base_dir_2, dst_base_dir):
    """
    合并两个放射组学特征表格文件，并更新 ID 列的命名以匹配新的文件名。
    """
    # 加载放射组学特征表
    csv_1 = os.path.join(src_base_dir_1, 'selected_features.csv')
    csv_2 = os.path.join(src_base_dir_2, 'selected_features.csv')
    
    df1 = pd.read_csv(csv_1)
    df2 = pd.read_csv(csv_2)

    # 为两张表格的 ID 添加前缀
    df1['ID'] = 'sm1_' + df1['ID'].astype(str)
    df2['ID'] = 'sm2_' + df2['ID'].astype(str)

    # 合并数据框
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # 保存合并后的放射组学特征表
    merged_csv_path = os.path.join(dst_base_dir, 'selected_features.csv')
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"合并后的放射组学特征表已保存为: {merged_csv_path}")

# 设置源文件夹和目标文件夹路径
src_dir_1 = '/home/yuwenjing/data/sm_1'
src_dir_2 = '/home/yuwenjing/data/sm_2'
dst_dir = '/home/yuwenjing/data/sm_3'

# 创建合并后的文件夹结构
create_folder_structure(dst_dir)

# 复制并重命名 sm_1 和 sm_2 中的文件
copy_and_rename_files(src_dir_1, dst_dir, 'sm1')
copy_and_rename_files(src_dir_2, dst_dir, 'sm2')

# 合并放射组学特征表格
merge_radiomic_features(src_dir_1, src_dir_2, dst_dir)

print("数据集合并完成！")
