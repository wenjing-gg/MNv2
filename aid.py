import numpy as np
import matplotlib.pyplot as plt
import nrrd

def analyze_ct_distribution(ct_image, percent=95):
    """
    分析 CT 图像数据分布，并根据数据分布推荐窗口范围。
    Args:
        ct_image (numpy.ndarray): CT 图像数据，形状可以是 2D 或 3D，例如 (H, W) 或 (D, H, W)。
        percent (int): 百分比范围，例如 95 表示推荐窗口范围覆盖 95% 的数据。
    Returns:
        tuple: 推荐的窗口范围 (window_min, window_max)。
    """
    # 确保输入为 NumPy 数组
    if not isinstance(ct_image, np.ndarray):
        raise TypeError("输入必须是 NumPy 数组")

    # 计算统计信息
    min_val = np.min(ct_image)
    max_val = np.max(ct_image)
    mean_val = np.mean(ct_image)
    std_val = np.std(ct_image)

    print(f"统计信息:")
    print(f"最小值 (Min): {min_val}")
    print(f"最大值 (Max): {max_val}")
    print(f"均值 (Mean): {mean_val:.2f}")
    print(f"标准差 (Std): {std_val:.2f}")

    # 计算推荐的窗口范围
    lower_bound = np.percentile(ct_image, (100 - percent) / 2)
    upper_bound = np.percentile(ct_image, 100 - (100 - percent) / 2)
    print(f"推荐的窗口范围覆盖 {percent}% 的数据: [{lower_bound:.2f}, {upper_bound:.2f}]")

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(ct_image.flatten(), bins=100, color='blue', alpha=0.7, label='HU Distribution')
    plt.axvline(lower_bound, color='red', linestyle='--', label=f'Lower Bound ({lower_bound:.2f})')
    plt.axvline(upper_bound, color='green', linestyle='--', label=f'Upper Bound ({upper_bound:.2f})')
    plt.title(f"CT 图像 HU 值分布及推荐范围 ({percent}%)")
    plt.xlabel("HU 值")
    plt.ylabel("频率")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    return lower_bound, upper_bound


if __name__ == "__main__":
    # 文件路径
    nrrd_file_path = "/home/yuwenjing/data/sm_3/test/0/sm1_2.nrrd"

    # 读取 NRRD 文件
    try:
        ct_image, header = nrrd.read(nrrd_file_path)
        print(f"成功加载 NRRD 文件: {nrrd_file_path}")
        print(f"图像形状: {ct_image.shape}")
    except Exception as e:
        print(f"读取 NRRD 文件时出错: {e}")
        exit()

    # 分析分布并推荐窗口范围
    recommended_min, recommended_max = analyze_ct_distribution(ct_image, percent=95)
    print(f"推荐的窗口范围: [{recommended_min:.2f}, {recommended_max:.2f}]")
