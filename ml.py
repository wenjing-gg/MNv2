# 导入必要的库
import os
import SimpleITK as sitk
import pandas as pd
import numpy as np
from radiomics import featureextractor
from sklearn.feature_selection import VarianceThreshold, RFECV, SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# 忽略警告信息
import warnings
warnings.filterwarnings('ignore')

# 设置数据路径
data_dir_non_metastasis = '/home/yuwenjing/data/肾母细胞瘤CT数据/Data3-0'  # 无转移数据
data_dir_metastasis = '/home/yuwenjing/data/肾母细胞瘤CT数据/Data3-1'     # 有转移数据

# 初始化列表以存储图像路径、掩膜路径和标签
image_paths = []
mask_paths = []
labels = []

# 处理无转移数据（标签为0）
for filename in os.listdir(data_dir_non_metastasis):
    if filename.endswith('_image.nrrd'):
        base_name = filename[:-11]  # 去掉'_image.nrrd'，得到基础名称
        image_path = os.path.join(data_dir_non_metastasis, filename)
        mask_filename = base_name + '_label.nrrd'
        mask_path = os.path.join(data_dir_non_metastasis, mask_filename)
        # 检查掩膜文件是否存在
        if os.path.exists(mask_path):
            image_paths.append(image_path)
            mask_paths.append(mask_path)
            labels.append(0)  # 无转移标签
        else:
            print(f"未找到图像 {image_path} 对应的掩膜文件。")

# 处理有转移数据（标签为1）
for filename in os.listdir(data_dir_metastasis):
    if filename.endswith('_image.nrrd'):
        base_name = filename[:-11]
        image_path = os.path.join(data_dir_metastasis, filename)
        mask_filename = base_name + '_label.nrrd'
        mask_path = os.path.join(data_dir_metastasis, mask_filename)
        if os.path.exists(mask_path):
            image_paths.append(image_path)
            mask_paths.append(mask_path)
            labels.append(1)  # 有转移标签
        else:
            print(f"未找到图像 {image_path} 对应的掩膜文件。")

# 打印收集到的样本数量
print(f"共收集到 {len(image_paths)} 个样本。")

# 初始化放射组学特征提取器
params = {}  # 如果有参数配置文件，可以在此加载，例如：params = 'params.yaml'
extractor = featureextractor.RadiomicsFeatureExtractor(**params)

# 存储所有样本的特征
all_features = []

# 循环读取图像和掩膜，提取特征
for idx, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
    # 读取图像和掩膜
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    # 提取特征
    features = extractor.execute(image, mask)
    # 将特征存储为字典，忽略诊断信息
    features_dict = {}
    for key, value in features.items():
        if 'diagnostics' not in key:
            features_dict[key] = value
    # 添加样本编号
    features_dict['PatientID'] = idx
    all_features.append(features_dict)

# 转换为DataFrame
features_df = pd.DataFrame(all_features)

# 将特征名中包含的'original_'去掉，简化特征名
features_df.columns = [col.replace('original_', '') for col in features_df.columns]

# 添加标签
features_df['Label'] = labels

# 将PatientID设置为索引
features_df.set_index('PatientID', inplace=True)

# 分离特征和标签
X = features_df.drop('Label', axis=1)
y = features_df['Label']

# 检查并处理缺失值
if X.isnull().values.any():
    print("检测到缺失值，用均值填充。")
    X.fillna(X.mean(), inplace=True)

# 将所有列转换为数值类型
X = X.apply(pd.to_numeric, errors='coerce')

# 再次检查缺失值
if X.isnull().values.any():
    print("转换后仍存在缺失值，用均值填充。")
    X.fillna(X.mean(), inplace=True)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def variance_threshold_selection(X, threshold=0.01):
    selector = VarianceThreshold(threshold=threshold)
    X_var = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support(indices=True)]
    return pd.DataFrame(X_var, columns=selected_features)

def lasso_selection(X, y, alpha=0.001, max_iter=10000, num_iterations=50, top_n_features=100):
    # 转换为 DataFrame 以保持列名
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
    else:
        X_df = X

    feature_counts = np.zeros(X_df.shape[1])
    
    for i in range(num_iterations):
        # 设置不同的随机种子
        X_train, _, y_train, _ = train_test_split(X_df, y, test_size=0.2, random_state=i, stratify=y)
        
        # 初始化 LASSO 模型
        lasso = Lasso(alpha=alpha, max_iter=max_iter).fit(X_train, y_train)
        selected_features_mask = lasso.coef_ != 0
        feature_counts += selected_features_mask  # 记录被选择特征的次数
    
    # 计算特征被选择的频率
    feature_frequency = feature_counts / num_iterations
    feature_frequency_series = pd.Series(feature_frequency, index=X_df.columns)
    
    # 选择频率最高的前 top_n_features 个特征
    stable_features = feature_frequency_series.nlargest(top_n_features).index
    
    print(f"指定选择的特征数量：{top_n_features}")
    print("选择的稳定特征列表：", stable_features.tolist())
    
    return pd.DataFrame(X_df[stable_features], columns=stable_features)

def rfecv_selection(X, y):
    estimator = RandomForestClassifier(random_state=42)
    selector = RFECV(estimator, step=1, cv=5, scoring='roc_auc', min_features_to_select=10)
    selector.fit(X, y)
    X_rfe = X.loc[:, selector.support_]
    return X_rfe

# 执行不同的特征选择方法
print("\n方差阈值法筛选：")
X_var = variance_threshold_selection(X, threshold=0.01)
print("选择的特征数量：", X_var.shape[1])

print("\nLASSO特征选择：")
X_lasso = lasso_selection(X_scaled, y)
print("选择的特征数量：", X_lasso.shape[1])

# RFECV耗时太久，不划算
# print("\nRFECV特征选择：")
# X_rfe = rfecv_selection(X, y)
# print("选择的特征数量：", X_rfe.shape[1])

# 比较不同方法的特征选择结果
feature_sets = {
    'Variance Threshold': X_var,
    'LASSO': X_lasso,
    # 'RFECV': X_rfe
}

# 将数据分割为训练集和测试集，并评估每种方法
cv = StratifiedKFold(5)
smote = SMOTE(random_state=42)

for method, X_selected in feature_sets.items():
    print(f"\n=== {method} 特征选择方法 ===")
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
    
    # 处理类别不平衡
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # 定义模型
    model = RandomForestClassifier(random_state=42)
    
    # 交叉验证评分
    scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')
    print("交叉验证准确率：", scores)
    print("平均准确率：", scores.mean())
    
    # 在测试集上评估
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    print("测试集分类报告：")
    print(classification_report(y_test, y_pred))
    print("测试集混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))
    
    # 计算AUC
    y_scores = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_scores)
    print(f"AUC值：{auc:.4f}")