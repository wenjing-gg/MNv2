import os
import random
import nrrd
import numpy as np
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split


def find_voi_bounds(label_data):
    """
    确定标签数据中 VOI 的边界。
    """
    non_zero_indices = np.argwhere(label_data)
    min_bounds = non_zero_indices.min(axis=0)
    max_bounds = non_zero_indices.max(axis=0)
    return min_bounds, max_bounds


def crop_image(image_data, min_bounds, max_bounds, target_size):
    """
    根据 VOI 边界裁剪图像，并调整为目标大小。
    """
    cropped_image = image_data[
        min_bounds[0]:max_bounds[0]+1,
        min_bounds[1]:max_bounds[1]+1,
        min_bounds[2]:max_bounds[2]+1
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


def process_and_save(data_dirs, label, output_dir, target_size=(128, 128, 128), split_ratio=0.8):
    """
    处理指定目录下的 NRRD 文件，裁剪并保存处理后的图像，同时划分为训练集和测试集。
    """
    output_train_dir = os.path.join(output_dir, 'train', str(label))
    output_test_dir = os.path.join(output_dir, 'test', str(label))
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)

    files = []
    for data_dir in data_dirs:
        files.extend([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('label.nrrd')])

    random.shuffle(files)
    train_count = int(len(files) * split_ratio)
    image_index = 1  # 初始化文件编号

    for i, label_path in enumerate(files):
        image_path = label_path.replace('label', 'image')
        
        if not os.path.exists(image_path):
            print(f"对应的图像文件不存在：{image_path}")
            continue

        # 读取标签和图像数据
        label_data, _ = nrrd.read(label_path)
        image_data, _ = nrrd.read(image_path)

        # 确定 VOI 边界
        min_bounds, max_bounds = find_voi_bounds(label_data)

        # 裁剪图像
        cropped_image = crop_image(image_data, min_bounds, max_bounds, target_size)

        # 保存裁剪后的图像，按顺序编号
        if i < train_count:
            output_path = os.path.join(output_train_dir, f"ID_{image_index:04d}.nrrd")
        else:
            output_path = os.path.join(output_test_dir, f"ID_{image_index:04d}.nrrd")

        nrrd.write(output_path, cropped_image)
        print(f"已保存裁剪后的图像：{output_path}")

        # 增加编号
        image_index += 1


def rename_nrrd_files(base_dirs):
    """
    重命名指定目录下的 nrrd 文件，仅使用唯一 ID 作为文件名，以便匹配放射组学特征。

    参数：
    - base_dirs: 包含四个路径的列表，每个路径中存储了待处理的 nrrd 文件。
    """
    id_count = 1  # 全局唯一 ID 计数器

    for base_dir in base_dirs:
        for filename in os.listdir(base_dir):
            if filename.endswith('.nrrd'):
                old_path = os.path.join(base_dir, filename)
                new_filename = f"ID_{id_count}.nrrd"
                new_path = os.path.join(base_dir, new_filename)

                # 重命名文件
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

                id_count += 1


def extract_radiomic_features(base_dirs, output_csv_path):
    """
    提取指定目录下的 nrrd 文件的放射组学特征，并使用 LASSO 特征选择保存最稳定的 100 个特征为 CSV 文件。
    """
    extractor = featureextractor.RadiomicsFeatureExtractor()
    all_features = []

    for base_dir in base_dirs:
        for filename in os.listdir(base_dir):
            if filename.endswith('.nrrd'):
                file_path = os.path.join(base_dir, filename)
                image = sitk.ReadImage(file_path)
                mask = sitk.ReadImage(file_path.replace('image', 'label'))
                features = extractor.execute(image, mask)
                features_dict = {key: value for key, value in features.items() if 'diagnostics' not in key}
                features_dict['ID'] = filename.split('.')[0]  # 使用文件名作为 ID
                all_features.append(features_dict)

    # 转换为 DataFrame
    features_df = pd.DataFrame(all_features)

    # 分离特征和标签
    X = features_df.drop(columns=['ID'])
    y = np.ones(X.shape[0])  # 使用虚拟标签（因为 LASSO 需要目标值）

    # 标准化特征
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用 LASSO 进行特征选择
    lasso = Lasso(alpha=0.001, max_iter=10000).fit(X_scaled, y)
    selected_features_mask = lasso.coef_ != 0
    feature_frequency = np.zeros(X.shape[1])

    # 多次运行 LASSO 获取稳定特征
    for i in range(50):
        X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=i)
        lasso.fit(X_train, y_train)
        selected_features_mask = lasso.coef_ != 0
        feature_frequency += selected_features_mask

    # 选择出现频率最高的前 100 个特征
    feature_frequency_series = pd.Series(feature_frequency, index=X.columns)
    stable_features = feature_frequency_series.nlargest(100).index

    # 保存最稳定的 100 个特征到 CSV
    selected_features_df = features_df[['ID'] + stable_features.tolist()]
    selected_features_df.to_csv(output_csv_path, index=False)
    print(f"已保存放射组学特征到 {output_csv_path}")


# 设置路径和目标文件夹
data_dirs_no_metastasis = ['/home/yuwenjing/data/肾母细胞瘤CT数据/Data0']
data_dirs_metastasis = ['/home/yuwenjing/data/肾母细胞瘤CT数据/Data1']
output_dir = '/home/yuwenjing/data/sm_1'

# 处理无转移数据（标签为0）
process_and_save(data_dirs_no_metastasis, label=0, output_dir=output_dir)

# 处理有转移数据（标签为1）
process_and_save(data_dirs_metastasis, label=1, output_dir=output_dir)

# 定义目录
train_0_dir = '/home/yuwenjing/data/sm_1/train/0/'
train_1_dir = '/home/yuwenjing/data/sm_1/train/1/'
test_0_dir = '/home/yuwenjing/data/sm_1/test/0/'
test_1_dir = '/home/yuwenjing/data/sm_1/test/1/'

# 调用函数进行重命名
rename_nrrd_files([train_0_dir, train_1_dir, test_0_dir, test_1_dir])

# 提取放射组学特征并保存为 CSV
extract_radiomic_features([train_0_dir, train_1_dir, test_0_dir, test_1_dir], '/home/yuwenjing/data/radiomic_features.csv')
