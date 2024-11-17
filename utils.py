import os
import sys
import json
import pickle
import random
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import torch
from tqdm import tqdm
from torch.cuda import amp

import matplotlib.pyplot as plt
import torch.cuda.amp as amp  # 引入混合精度训练所需的模块


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    criterion = nn.CrossEntropyLoss()

    # 添加进度条显示
    data_loader = tqdm(data_loader, file=sys.stdout)
    data_loader.set_description(f"Training Epoch {epoch}")

    # 实际样本数的累积
    total_samples = 0

    for batch_idx, batch in enumerate(data_loader):
        imgs, labels, radiomics = batch
        imgs = imgs.to(device)
        labels = labels.to(device)
        radiomics = radiomics.to(device)

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(imgs, radiomics)
                print(outputs)
                print(labels)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs, radiomics)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * imgs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        # 累积样本数
        total_samples += imgs.size(0)

        # 计算当前的平均损失和准确率
        current_loss = running_loss / total_samples
        current_acc = running_corrects.double() / total_samples

        # 实时更新进度条描述
        data_loader.set_description(f"Training Epoch {epoch} - Loss: {current_loss:.4f}, Acc: {current_acc:.4f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples

    return epoch_loss, epoch_acc.item()


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, num_classes=2):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    all_probs = []

    # 添加进度条显示
    data_loader = tqdm(data_loader, file=sys.stdout)
    data_loader.set_description(f"Evaluating Epoch {epoch}")

    # 实际样本数的累积
    total_samples = 0

    for batch_idx, batch in enumerate(data_loader):
        imgs, labels, radiomics = batch
        imgs = imgs.to(device)
        labels = labels.to(device)
        radiomics = radiomics.to(device)

        outputs = model(imgs, radiomics)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        probs = nn.functional.softmax(outputs, dim=1)[:, 1]  # 取第2类的概率作为预测值

        running_loss += loss.item() * imgs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        # 累积样本数
        total_samples += imgs.size(0)

        # 计算当前的平均损失和准确率
        current_loss = running_loss / total_samples
        current_acc = running_corrects.double() / total_samples

        # 实时更新进度条描述
        data_loader.set_description(f"Evaluating Epoch {epoch} - Loss: {current_loss:.4f}, Acc: {current_acc:.4f}")

    # 计算epoch的平均损失和准确率
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    auc = roc_auc_score(all_labels, all_probs)  # 计算AUC值

    # 计算混淆矩阵并从中提取指标
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 防止除以零
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # 防止除以零

    print("\n测试集分类报告：")
    print(classification_report(all_labels, all_preds))
    print("测试集混淆矩阵：")
    print(confusion_matrix(all_labels, all_preds))
    print(f"AUC值：{auc:.4f}")
    print(f"ACC值：{epoch_acc:.4f}")
    print(f"敏感性（Sensitivity）: {sensitivity:.4f}")
    print(f"特异性（Specificity）: {specificity:.4f}\n")

    return epoch_loss, epoch_acc.item(), auc, sensitivity, specificity  # 返回所有指标

