import os
import time
import argparse
import sys
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch_optimizer import RAdam
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyNRRDDataSet  # 使用您自定义的数据集类
from sup import SwinUNETRClassifier as create_model  # 确保模型为3D版本
from utils import train_one_epoch, evaluate
from torch.cuda.amp import GradScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def wait_for_available_gpu():
    """依次检查显卡是否空闲，一旦找到空闲的显卡就返回其ID"""
    while True:
        for device_id in range(4):  # 检查cuda:0到cuda:3
            # 获取显卡的空闲和总显存信息
            free_mem, total_mem = torch.cuda.mem_get_info(device_id)
            
            if free_mem >= 0.9*total_mem:  # 如果空闲显存等于总显存，表示显卡未被占用
                print(f"Device cuda:{device_id} is now fully available. Starting training...")
                return device_id  # 返回找到的空闲设备ID
            
            else:
                print(f"Device cuda:{device_id} is currently in use. Free memory: {free_mem} bytes, Total memory: {total_mem} bytes.")
        
        # 如果没有找到空闲显卡，等待 60 秒后重新检查
        print("No available GPU found. Waiting...")
        time.sleep(60)

def main(args):
    # 等待并选择一个空闲的显卡
    available_device_id = wait_for_available_gpu()
    # 设置设备
    device = torch.device(f"cuda:{available_device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device} for training.")
    
    # 初始化混合精度梯度缩放器
    scaler = GradScaler() if args.use_amp else None

    # 创建保存模型的目录
    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    # 初始化 TensorBoard 写入器
    tb_writer = SummaryWriter()
    # 实例化训练数据集
    train_dataset = MyNRRDDataSet(
        root_dir=args.data_path,
        split='train',
        radiomic_csv=args.radiomic_csv,  # 确保传递 radiomic_csv 参数
        target_shape=(64,64,64),  # 修改后的目标形状
        num_augmentations=args.num_augmentations
    )

    # 实例化验证数据集,验证集无需数据增强
    val_dataset = MyNRRDDataSet(
        root_dir=args.data_path,
        split='test',
        radiomic_csv=args.radiomic_csv,  # 确保传递 radiomic_csv 参数
        target_shape=(64,64,64),  # 修改后的目标形状
        num_augmentations=0
    )

    # 定义数据加载器
    batch_size = args.batch_size
    nw = 8
    print(f'Using {nw} dataloader workers every process')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn
    )

    # 初始化模型，设置 radiomics_size
    # 确保 radiomics_size 与选定的特征数量一致
    if len(train_dataset.data_list) > 0:
        radiomics_size = train_dataset.data_list[0][2].shape[0]  # 假设所有样本特征数相同
    else:
        raise ValueError("训练数据集为空。请检查数据加载。")
    print(f"Radiomics Size: {radiomics_size}")
    
    model = create_model(
        img_size=(128, 128, 128),       # 根据输入尺寸调整
        in_channels=1,                   # 输入通道数
        num_classes=args.num_classes,    # 分类数
        radiomics_size=radiomics_size,    # radiomics_size
        feature_size=48,                 # 根据需要调整特征大小
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,
        spatial_dims=3,
        norm_name="instance",
    ).to(device)
    print("Model's state_dict keys:")
    for key in model.state_dict().keys():
        print(key)

    # 载入预训练权重（如果提供）
    if args.weights != "":
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' does not exist."
        checkpoint = torch.load(args.weights, map_location=device)
        
        # 提取权重字典
        if 'net' in checkpoint:
            weights_dict = checkpoint['net']
        else:
            weights_dict = checkpoint

        # 移除 `module.` 前缀
        weights_dict = {k.replace('module.', ''): v for k, v in weights_dict.items()}

        # 移除 `backbone.` 前缀（如果存在）
        weights_dict = {k.replace('backbone.', ''): v for k, v in weights_dict.items()}

        # 删除解码器相关的键
        decoder_keys = [k for k in weights_dict.keys() if k.startswith('decoder')]
        for k in decoder_keys:
            del weights_dict[k]

        # 处理输入通道数不匹配的问题
        conv1_key = 'swinViT.patch_embed.proj.weight'
        if conv1_key in weights_dict and weights_dict[conv1_key].shape[1] != model.swinViT.patch_embed.proj.weight.shape[1]:
            if weights_dict[conv1_key].shape[1] == 3 and model.swinViT.patch_embed.proj.weight.shape[1] == 1:
                # 将预训练权重在通道维度上取平均
                weights_dict[conv1_key] = weights_dict[conv1_key].mean(dim=1, keepdim=True)

        # 加载预训练权重
        load_info = model.load_state_dict(weights_dict, strict=False)

        # 打印加载信息
        print("Successfully loaded pre-trained weights.")
        print(f"Missing keys: {load_info.missing_keys}")
        print(f"Unexpected keys: {load_info.unexpected_keys}")

        # 计算加载的参数比例
        loaded_params = len(model.state_dict()) - len(load_info.missing_keys)
        total_params = len(model.state_dict())
        load_percentage = (loaded_params / total_params) * 100
        print(f"Percentage of loaded weights: {load_percentage:.2f}%")

    # 冻结部分层（如果指定）
    if args.freeze_layers:
        layers_to_train = ["swinViT.layers4","encoder10","global_pool","classifier"]  # 表现较好 auc=0.7126 acc=0.6829
        # layers_to_train = ["swinViT.layers3","swinViT.layers4","encoder3","encoder4","encoder10","classifier"]  

        for name, param in model.named_parameters():
            if any(layer in name for layer in layers_to_train):
                param.requires_grad = True  # 保持可训练状态
                print(f"Training {name}")
            else:
                param.requires_grad = False  # 冻结层
                print(f"Freezing {name}")

    # 定义优化器
    optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # 初始化学习率调度器，使用 'max' 模式，因为 AUC 越大越好
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # max 模式表示 AUC 不再上升时调整学习率
        factor=0.5,
        patience=8
    )

    # 训练参数
    num_epochs = args.epochs
    best_val_auc = 0
    early_stopping_patience = 50
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # 训练一个 epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            scaler=scaler
        )

        # 验证
        val_loss, val_acc, val_auc, val_sen, val_spe = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch,
            num_classes=args.num_classes
        )

        # 记录到 TensorBoard
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # 调整学习率
        scheduler.step(val_auc)

        # 检查是否是最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc4auc = val_acc
            best_val_sen4auc = val_sen
            best_val_spe4auc = val_spe
            torch.save(model.state_dict(), os.path.join("./weights", "best_model.pth"))
            print(f"Best model saved at epoch {epoch} with Val Auc={val_auc:.4f}")
            epochs_no_improve = 0
            best_epoch = epoch
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        # 早停检查
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {early_stopping_patience} epochs with no improvement.")
            break

        # 可选：打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join("./weights", "final_model.pth"))
    print("Training complete. Final model saved.")
    # 打印并保存最佳结果信息
    print(f"\nBest Results at Epoch {best_epoch}:")
    print(f"Best Val AUC: {best_val_auc:.4f}")
    print(f"Best Val Acc: {best_val_acc4auc:.4f}")
    print(f"Best Val Sensitivity: {best_val_sen4auc:.4f}")
    print(f"Best Val Specificity: {best_val_spe4auc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Vision Transformer for 3D Classification')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of target classes')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=3e-4, help='Initial learning rate')
    
    # 数据集所在根目录
    parser.add_argument('--data_path', type=str, default="/home/yuwenjing/data/sm_3", help='Path to the dataset')

    # 放射组学特征CSV文件路径
    parser.add_argument('--radiomic_csv', type=str, default='/home/yuwenjing/data/sm_3/selected_features_.csv', help='Path to the radiomic features CSV file')
    
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='/home/yuwenjing/DeepLearning/MN/supervised_suprem_swinunetr_2100.pth', help='Initial weights path')
    
    # 是否冻结权重
    parser.add_argument('--freeze_layers', type=bool, default=True, help='Freeze layers except head and pre_logits')
    
    # 设备选择
    parser.add_argument('--device', type=str, default='cuda:3', help='Device ID (e.g., cuda:0 or cpu)')
    
    # 是否使用混合精度训练
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision')
    
    # 数据增强数量
    parser.add_argument('--num_augmentations', type=int, default=8, help='Number of augmentations per sample during training')
    
    opt = parser.parse_args()
    
    main(opt)