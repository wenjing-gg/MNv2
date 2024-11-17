# 代码使用简介

本代码用于 CT 图像的二分类任务（使用 Swin UNETR 的预训练权重进行迁移学习）。

## 使用步骤

1. **下载预训练权重**
   - 从官方或指定来源下载 Swin UNETR 的预训练权重。https://share.weiyun.com/QsHDRqw5

2. **下载数据集**
   - 获取并下载所需的 CT 图像数据集。

3. **安装依赖**
   - 执行以下命令安装必要的依赖库：
     ```bash
     pip install -r requirements.txt
     ```

4. **预处理原始数据**
   - 运行以下脚本处理原始数据：
     ```bash
     python process_rawcsv.py
     ```

5. **启动训练**
   - 执行训练脚本：
     ```bash
     python train_sup.py
     ```
