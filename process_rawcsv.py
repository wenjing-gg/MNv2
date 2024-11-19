#cen1
import os
import nrrd
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # 用于均匀划分训练集和测试集

def find_voi_bounds(label_data):
    """
    确定标签数据中 VOI 的边界。
    """
    non_zero_indices = np.argwhere(label_data)
    if non_zero_indices.size == 0:
        return None, None  # 没有 VOI 区域时返回 None
    min_bounds = non_zero_indices.min(axis=0)
    max_bounds = non_zero_indices.max(axis=0)
    return min_bounds, max_bounds

def crop_image(image_data, min_bounds, max_bounds, target_size, expansion_factor=1.2):
    """
    根据 VOI 边界裁剪图像，扩展裁剪范围，并调整为目标大小。
    """
    if min_bounds is None or max_bounds is None:
        return image_data  # 如果没有 VOI 区域，则不进行裁剪

    # 计算扩展后的边界范围
    center = [(min_bound + max_bound) / 2 for min_bound, max_bound in zip(min_bounds, max_bounds)]
    half_size = [(max_bound - min_bound) / 2 * expansion_factor for min_bound, max_bound in zip(min_bounds, max_bounds)]
    
    # 扩展后的 min 和 max bounds
    new_min_bounds = [int(max(0, center[i] - half_size[i])) for i in range(3)]
    new_max_bounds = [int(min(image_data.shape[i] - 1, center[i] + half_size[i])) for i in range(3)]

    # 裁剪图像
    cropped_image = image_data[
        new_min_bounds[0]:new_max_bounds[0]+1,
        new_min_bounds[1]:new_max_bounds[1]+1,
        new_min_bounds[2]:new_max_bounds[2]+1
    ]

    # 使用 SimpleITK 进行重采样
    sitk_image = sitk.GetImageFromArray(np.transpose(cropped_image, (2, 1, 0)))
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    new_spacing = [
        original_spacing[0] * (original_size[0] / target_size[0]),
        original_spacing[1] * (original_size[1] / target_size[1]),
        original_spacing[2] * (original_size[2] / target_size[2])
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(target_size)
    resample.SetInterpolator(sitk.sitkLinear)
    resampled_image = resample.Execute(sitk_image)
    return np.transpose(sitk.GetArrayFromImage(resampled_image), (2, 1, 0))


def extract_radiomic_features_from_xlsx(data_dirs):
    """
    从多个 Excel 文件中提取放射组学特征，合并所有特征并返回。
    """
    all_features = []

    for data_dir in data_dirs:
        radiomic_file = os.path.join(data_dir, f"results{data_dir[-1]}-hz.xlsx")
        print(f"读取放射组学特征文件: {radiomic_file}")

        if not os.path.exists(radiomic_file):
            print(f"警告: 放射组学特征文件不存在：{radiomic_file}")
            continue

        # 读取放射组学特征的 Excel 文件
        df = pd.read_excel(radiomic_file)

        # 添加标签列，0表示无转移，1表示有转移
        label = 0 if 'Data0' in data_dir else 1
        df['Label'] = label

        # 从 Unique_ID 中提取对应的 ID（去掉 "_image"）
        df['ID'] = df['Unique_ID'].apply(lambda x: x)  # 保留原始格式

        # 创建一个字典，用于快速查找特征
        radiomic_features_dict = df.set_index('ID').to_dict('index')

        # 遍历当前数据文件夹，提取与之对应的放射组学特征
        for filename in os.listdir(data_dir):
            if filename.endswith('.nrrd') and '_image.nrrd' in filename:
                # 提取图像文件名中的 ID 部分
                unique_id = filename.replace('_image.nrrd', '_image')  # 确保 ID 格式是 "105_image"
                
                if unique_id in radiomic_features_dict:
                    features_dict = radiomic_features_dict[unique_id]
                    features_dict['ID'] = unique_id  # 使用与 CSV 中一致的 ID
                    all_features.append(features_dict)
                else:
                    print(f"Warning: Radiomic features for ID {unique_id} not found in Excel file.")

    return all_features


def lasso_feature_selection(features_df, num_features=100):
    """
    使用 Lasso 回归选择最稳定的特征。
    """
    X = features_df.drop(columns=['Unique_ID', 'Label'])
    y = features_df['Label']

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Lasso 回归模型
    lasso = Lasso(alpha=0.01)  # 调整 alpha 参数
    lasso.fit(X_scaled, y)

    # 获取系数大的特征
    feature_importance = np.abs(lasso.coef_)
    selected_indices = feature_importance.argsort()[-num_features:]

    selected_features = X.columns[selected_indices]
    print(f"选出的 {num_features} 个稳定特征: {selected_features}")
    return selected_features


def process_and_crop_images(data_dirs, output_dir, target_size=(128,128,128)):
    """
    处理指定目录下的 NRRD 文件，裁剪图像并保存裁剪后的图像（不进行数据集划分）。
    """
    # 确保保存数据的目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 如果目标目录不存在，创建目录
        
    # 提取放射组学特征
    radiomic_features = extract_radiomic_features_from_xlsx(data_dirs)

    # 将特征字典转换为 DataFrame 方便后续操作
    radiomic_features_df = pd.DataFrame(radiomic_features)
    
    # 确保 DataFrame 中有 'ID' 列
    if 'ID' not in radiomic_features_df.columns:
        print("错误: 放射组学特征数据中缺少 'ID' 列!")
        return

    radiomic_features_df.set_index('ID', inplace=True)

    # 使用 Lasso 选择最稳定的 100 个特征
    selected_features = lasso_feature_selection(radiomic_features_df)

    # 选出的特征保存为 CSV 文件
    selected_features_df = radiomic_features_df[selected_features]
    selected_features_df.to_csv(os.path.join(output_dir, 'selected_features.csv'))
    print(f"选出的放射组学特征已保存为：{os.path.join(output_dir, 'selected_features.csv')}")

    # 统一裁剪图像
    cropped_images = []

    for i, (unique_id, features) in enumerate(radiomic_features_df.iterrows()):
        # 打印unique_id和features中的Label列
        print(f"unique_id: {unique_id}, Label: {features['Label']}")
        
        # 查找图像文件和标签文件
        image_path = None
        label_path = None

        # 删掉unique_id后面的 _image
        unique_id = unique_id.replace('_image', '')
        for data_dir in data_dirs:
            possible_image_path = os.path.join(data_dir, f"{unique_id}_image.nrrd")
            possible_label_path = os.path.join(data_dir, f"{unique_id}_label.nrrd")

            if os.path.exists(possible_image_path):
                image_path = possible_image_path
            if os.path.exists(possible_label_path):
                label_path = possible_label_path

            # 如果都找到了，就可以停止查找
            if image_path and label_path:
                break

        if not image_path or not label_path:
            print(f"警告: 图像或标签文件未找到，ID: {unique_id}")
            continue

        # 获取图像和标签数据
        label_data, _ = nrrd.read(label_path)
        image_data, _ = nrrd.read(image_path)
        print(f"处理图像：{image_path}")

        # 确定 VOI 边界
        min_bounds, max_bounds = find_voi_bounds(label_data)

        # 裁剪图像
        cropped_image = crop_image(image_data, min_bounds, max_bounds, target_size)
        cropped_images.append((unique_id, cropped_image, features['Label']))

    return cropped_images


def split_and_save_data(cropped_images, output_dir, split_ratio=0.8):
    """
    根据 split_ratio 划分训练集和测试集，并保存裁剪后的图像。
    确保标签0和标签1均匀划分。
    """
    # 根据标签进行划分，保证训练集和测试集标签分布均匀
    label_0_images = [image for image in cropped_images if image[2] == 0]
    label_1_images = [image for image in cropped_images if image[2] == 1]

    # 按 8:2 划分训练集和测试集
    label_0_train, label_0_test = train_test_split(label_0_images, test_size=0.2, random_state=42)
    label_1_train, label_1_test = train_test_split(label_1_images, test_size=0.2, random_state=42)

    # 合并训练集和测试集
    train_images = label_0_train + label_1_train
    test_images = label_0_test + label_1_test

    # 创建存储路径
    output_train_dir = os.path.join(output_dir, 'train')
    output_test_dir = os.path.join(output_dir, 'test')
    
    # 创建类别文件夹
    os.makedirs(os.path.join(output_train_dir, '0'), exist_ok=True)
    os.makedirs(os.path.join(output_train_dir, '1'), exist_ok=True)
    os.makedirs(os.path.join(output_test_dir, '0'), exist_ok=True)
    os.makedirs(os.path.join(output_test_dir, '1'), exist_ok=True)

    # 保存训练集和测试集图像
    for image in train_images:
        unique_id, cropped_image, label = image
        output_path = os.path.join(output_train_dir, str(label), f"{unique_id}.nrrd")
        nrrd.write(output_path, cropped_image)
        print(f"已保存训练集图像：{output_path}")

    for image in test_images:
        unique_id, cropped_image, label = image
        output_path = os.path.join(output_test_dir, str(label), f"{unique_id}.nrrd")
        nrrd.write(output_path, cropped_image)
        print(f"已保存测试集图像：{output_path}")


# 设置路径和目标文件夹
data_dirs = ['/home/yuwenjing/data/肾母细胞瘤CT数据/Data0', '/home/yuwenjing/data/肾母细胞瘤CT数据/Data1']
output_dir = '/home/yuwenjing/data/sm_1'

# 裁剪所有图像并保存
cropped_images = process_and_crop_images(data_dirs, output_dir)

# 根据划分比例将图像分为训练集和测试集并保存
split_and_save_data(cropped_images, output_dir)


#cen2
import os
import nrrd
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # 用于均匀划分训练集和测试集

def find_voi_bounds(label_data):
    """
    确定标签数据中 VOI 的边界。
    """
    non_zero_indices = np.argwhere(label_data)
    if non_zero_indices.size == 0:
        return None, None  # 没有 VOI 区域时返回 None
    min_bounds = non_zero_indices.min(axis=0)
    max_bounds = non_zero_indices.max(axis=0)
    return min_bounds, max_bounds

def crop_image(image_data, min_bounds, max_bounds, target_size, expansion_factor=1.2):
    """
    根据 VOI 边界裁剪图像，扩展裁剪范围，并调整为目标大小。
    """
    if min_bounds is None or max_bounds is None:
        return image_data  # 如果没有 VOI 区域，则不进行裁剪

    # 计算扩展后的边界范围
    center = [(min_bound + max_bound) / 2 for min_bound, max_bound in zip(min_bounds, max_bounds)]
    half_size = [(max_bound - min_bound) / 2 * expansion_factor for min_bound, max_bound in zip(min_bounds, max_bounds)]
    
    # 扩展后的 min 和 max bounds
    new_min_bounds = [int(max(0, center[i] - half_size[i])) for i in range(3)]
    new_max_bounds = [int(min(image_data.shape[i] - 1, center[i] + half_size[i])) for i in range(3)]

    # 裁剪图像
    cropped_image = image_data[
        new_min_bounds[0]:new_max_bounds[0]+1,
        new_min_bounds[1]:new_max_bounds[1]+1,
        new_min_bounds[2]:new_max_bounds[2]+1
    ]

    # 使用 SimpleITK 进行重采样
    sitk_image = sitk.GetImageFromArray(np.transpose(cropped_image, (2, 1, 0)))
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    new_spacing = [
        original_spacing[0] * (original_size[0] / target_size[0]),
        original_spacing[1] * (original_size[1] / target_size[1]),
        original_spacing[2] * (original_size[2] / target_size[2])
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(target_size)
    resample.SetInterpolator(sitk.sitkLinear)
    resampled_image = resample.Execute(sitk_image)
    return np.transpose(sitk.GetArrayFromImage(resampled_image), (2, 1, 0))


def extract_radiomic_features_from_xlsx(data_dirs):
    """
    从多个 Excel 文件中提取放射组学特征，并根据文件夹中的文件顺序添加 Unique_ID 列。
    """
    all_features = []

    for data_dir in data_dirs:
        radiomic_file = os.path.join(data_dir, f"results{data_dir[-1]}-sz.xlsx")
        print(f"读取放射组学特征文件: {radiomic_file}")

        if not os.path.exists(radiomic_file):
            print(f"警告: 放射组学特征文件不存在：{radiomic_file}")
            continue

        # 读取放射组学特征的 Excel 文件
        df = pd.read_excel(radiomic_file)

        # 添加标签列，0表示无转移，1表示有转移
        label = 0 if 'Data3-0' in data_dir else 1
        df['Label'] = label

        # 根据文件夹中的文件顺序生成 Unique_ID 列
        nrrd_files = [f for f in sorted(os.listdir(data_dir)) if f.endswith('.nrrd') and '_image.nrrd' in f]
        df['Unique_ID'] = [file.replace('_image.nrrd', '_image') for file in nrrd_files]

        # 创建字典以便快速查找特征
        radiomic_features_dict = df.set_index('Unique_ID').to_dict('index')

        # 遍历 radiomic_features_dict 中的每一行，提取其 ID 和对应的特征
        for unique_id in radiomic_features_dict.keys():
            features_dict = radiomic_features_dict[unique_id]
            features_dict['ID'] = unique_id  # 确保 ID 格式与 CSV 格式一致
            all_features.append(features_dict)

    return all_features


def lasso_feature_selection(features_df, num_features=100):
    """
    使用 Lasso 回归选择最稳定的特征。
    """
    X = features_df.drop(columns=['Label'])
    y = features_df['Label']

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Lasso 回归模型
    lasso = Lasso(alpha=0.01)  # 调整 alpha 参数
    lasso.fit(X_scaled, y)

    # 获取系数大的特征
    feature_importance = np.abs(lasso.coef_)
    selected_indices = feature_importance.argsort()[-num_features:]

    selected_features = X.columns[selected_indices]
    print(f"选出的 {num_features} 个稳定特征: {selected_features}")
    return selected_features


def process_and_crop_images(data_dirs, output_dir, target_size=(128,128,128)):
    """
    处理指定目录下的 NRRD 文件，裁剪图像并保存裁剪后的图像（不进行数据集划分）。
    """
    # 确保保存数据的目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 如果目标目录不存在，创建目录
        
    # 提取放射组学特征
    radiomic_features = extract_radiomic_features_from_xlsx(data_dirs)

    # 将特征字典转换为 DataFrame 方便后续操作
    radiomic_features_df = pd.DataFrame(radiomic_features)
    
    # 确保 DataFrame 中有 'ID' 列
    if 'ID' not in radiomic_features_df.columns:
        print("错误: 放射组学特征数据中缺少 'ID' 列!")
        return

    radiomic_features_df.set_index('ID', inplace=True)

    # 使用 Lasso 选择最稳定的 100 个特征
    selected_features = lasso_feature_selection(radiomic_features_df)

    # 选出的特征保存为 CSV 文件
    selected_features_df = radiomic_features_df[selected_features]
    selected_features_df.to_csv(os.path.join(output_dir, 'selected_features.csv'))
    print(f"选出的放射组学特征已保存为：{os.path.join(output_dir, 'selected_features.csv')}")

    # 统一裁剪图像
    cropped_images = []

    for i, (unique_id, features) in enumerate(radiomic_features_df.iterrows()):
        # 打印unique_id和features中的Label列
        print(f"unique_id: {unique_id}, Label: {features['Label']}")
        
        # 查找图像文件和标签文件
        image_path = None
        label_path = None

        # 删掉unique_id后面的 _image
        unique_id = unique_id.replace('_image', '')
        for data_dir in data_dirs:
            possible_image_path = os.path.join(data_dir, f"{unique_id}_image.nrrd")
            possible_label_path = os.path.join(data_dir, f"{unique_id}_label.nrrd")

            if os.path.exists(possible_image_path):
                image_path = possible_image_path
            if os.path.exists(possible_label_path):
                label_path = possible_label_path

            # 如果都找到了，就可以停止查找
            if image_path and label_path:
                break

        if not image_path or not label_path:
            print(f"警告: 图像或标签文件未找到，ID: {unique_id}")
            continue

        # 获取图像和标签数据
        label_data, _ = nrrd.read(label_path)
        image_data, _ = nrrd.read(image_path)
        print(f"处理图像：{image_path}")

        # 确定 VOI 边界
        min_bounds, max_bounds = find_voi_bounds(label_data)

        # 裁剪图像
        cropped_image = crop_image(image_data, min_bounds, max_bounds, target_size)
        cropped_images.append((unique_id, cropped_image, features['Label']))

    return cropped_images


def split_and_save_data(cropped_images, output_dir, split_ratio=0.8):
    """
    根据 split_ratio 划分训练集和测试集，并保存裁剪后的图像。
    确保标签0和标签1均匀划分。
    """
    # 根据标签进行划分，保证训练集和测试集标签分布均匀
    label_0_images = [image for image in cropped_images if image[2] == 0]
    label_1_images = [image for image in cropped_images if image[2] == 1]

    # 按 8:2 划分训练集和测试集
    label_0_train, label_0_test = train_test_split(label_0_images, test_size=0.2, random_state=42)
    label_1_train, label_1_test = train_test_split(label_1_images, test_size=0.2, random_state=42)

    # 合并训练集和测试集
    train_images = label_0_train + label_1_train
    test_images = label_0_test + label_1_test

    # 创建存储路径
    output_train_dir = os.path.join(output_dir, 'train')
    output_test_dir = os.path.join(output_dir, 'test')
    
    # 创建类别文件夹
    os.makedirs(os.path.join(output_train_dir, '0'), exist_ok=True)
    os.makedirs(os.path.join(output_train_dir, '1'), exist_ok=True)
    os.makedirs(os.path.join(output_test_dir, '0'), exist_ok=True)
    os.makedirs(os.path.join(output_test_dir, '1'), exist_ok=True)

    # 保存训练集和测试集图像
    for image in train_images:
        unique_id, cropped_image, label = image
        output_path = os.path.join(output_train_dir, str(int(label)), f"{unique_id}.nrrd")
        nrrd.write(output_path, cropped_image)
        print(f"已保存训练集图像：{output_path}")

    for image in test_images:
        unique_id, cropped_image, label = image
        output_path = os.path.join(output_test_dir, str(int(label)), f"{unique_id}.nrrd")
        nrrd.write(output_path, cropped_image)
        print(f"已保存测试集图像：{output_path}")



# 设置路径和目标文件夹
data_dirs = ['/home/yuwenjing/data/肾母细胞瘤CT数据/Data3-0', '/home/yuwenjing/data/肾母细胞瘤CT数据/Data3-1']
output_dir = '/home/yuwenjing/data/sm_2'

# 裁剪所有图像并保存
cropped_images = process_and_crop_images(data_dirs, output_dir)

# 根据划分比例将图像分为训练集和测试集并保存
split_and_save_data(cropped_images, output_dir)



#combine
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

#clean
import pandas as pd
# Define the file path
file_path = '/home/yuwenjing/data/肾母细胞瘤CT数据/sm_3/selected_features.csv'
# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)
# Drop columns containing NaN values
df_cleaned = df.dropna(axis=1)
# Save the cleaned DataFrame back to a CSV file
output_path = '/home/yuwenjing/data/肾母细胞瘤CT数据/sm_3/selected_features_.csv'
df_cleaned.to_csv(output_path, index=False)
output_path
