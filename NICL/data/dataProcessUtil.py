import numpy as np
import torch
from torch.utils.data import random_split, Dataset

def get_grouplist(dataframe, grouplabel, sortlabel):
    groups = dataframe.groupby(grouplabel)
    group_list = []
    for grouplabel, group in groups:
        group = group.sort_values(by=sortlabel)
        group_list.append(group)
    return group_list

def createData(data_list):
    input1_list = []
    input2_list = []
    label_list = []
    for data in data_list:
        np.random.shuffle(data)
        for sat in data:
            input1_list.append(torch.tensor(sat[:-1], dtype=torch.float32))
            input2_list.append(torch.tensor(data[:,:-1], dtype=torch.float32))
            label_list.append(torch.tensor(sat[-1], dtype=torch.float32))
    dataset = MyDataset(input1_list, input2_list, label_list)
    return dataset

class MyDataset(Dataset):
    def __init__(self, data1, data2, labels):
        self.data1 = data1
        self.data2 = data2
        self.labels = labels

    def __getitem__(self, index):
        data1, data2, label = self.data1[index], self.data2[index], self.labels[index]
        return data1, data2, label

    def __len__(self):
        return len(self.data1)

def data_split(dataset,train_ratio):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

