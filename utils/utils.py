import os
import pickle
import shutil

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader

def myDataLoader(dataset, batch_size):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=lambda batch: (
        pad_sequence([item[0] for item in batch], batch_first=True),  # input1
        pack_padded_sequence(pad_sequence([item[1] for item in batch], batch_first=True), lengths=[item[1].size(0) for item in batch], batch_first=True, enforce_sorted=False),  # input2
        torch.stack([item[2] for item in batch])))  # labels
    return dataloader


def get_correct(predicted_labels,true_labels):
    # 将预测的标签转换为类别
    predicted_classes = torch.argmax(predicted_labels, dim=1)
    # 将真实的标签转换为类别
    true_classes = torch.argmax(true_labels, dim=1)
    # 比较预测的类别和真实的类别，得到一个布尔张量
    correct_predictions = torch.eq(predicted_classes, true_classes)

    # 统计正确的预测个数
    correct = torch.sum(correct_predictions).item()
    return correct

def makedir(ROOT, LR, BATCH_SIZE, N_EPOCH, info):
    # 创建文件夹1
    folder1_name = 'Train_LR' + str(LR) + '_batch'+ str(BATCH_SIZE) + '_epoch' + str(N_EPOCH) +  '_adam_' + str(info)
    folder1_path = os.path.join(ROOT, folder1_name)
    # 如果文件夹已经存在，则删除它
    if os.path.exists(folder1_path):
        shutil.rmtree(folder1_path)
    os.makedirs(folder1_path)

    # 创建文件夹2
    folder2_name = 'model_epoch_ir=' + str(LR)
    folder2_path = os.path.join(folder1_path, folder2_name)
    if os.path.exists(folder2_path):
        shutil.rmtree(folder2_path)
    os.makedirs(folder2_path)
    return folder1_path, folder2_path

def pltshow1(list1, LR, BATCH_SIZE, N_EPOCHS, folder1_path):
    data1 = np.array(list1)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # 绘制训练损失和验证损失曲线
    ax1.plot(data1, label=' Loss', color='green')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')
    # 调整图表布局
    fig.tight_layout()
    # 保存到文件位置
    plot_name = str(LR) + '_' + str(BATCH_SIZE) + '_' + str(N_EPOCHS)  + '_plot.png'
    plt.savefig(os.path.join(folder1_path, plot_name))
    # 显示图表
    plt.show()
    # 关闭图形对象


# def pltshow(list1, list2, LR, BATCH_SIZE, N_EPOCHS, folder1_path):
#     data1 = np.array(list1)
#     data2 = np.array(list2)
#
#     fig, ax1 = plt.subplots(figsize=(10, 6))
#
#     # 绘制训练损失和验证损失曲线
#     ax1.plot(data1, label=' Loss', color='green')
#     ax1.plot(data2, label=' acc', color='blue')
#     ax1.set_xlabel('Epochs')
#     ax1.set_ylabel('Loss and acc')
#     ax1.legend(loc='upper left')
#
#     # 调整图表布局
#     fig.tight_layout()
#     # 保存到文件位置
#     plot_name = str(LR) + '_' + str(BATCH_SIZE) + '_' + str(N_EPOCHS)  + '_plot.png'
#     plt.savefig(os.path.join(folder1_path, plot_name))
#     # 显示图表
#     plt.show()
#     # 关闭图形对象