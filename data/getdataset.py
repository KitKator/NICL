import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import random_split, Dataset



def openpickle(pickle_name):
    with open(pickle_name, 'rb') as f:
        Pdata = pickle.load(f)

    Pdata['LOS/NLOS_label'].replace(-1.0, 0, inplace=True)
    Pdata = Pdata[Pdata['Pr_rate_consitency'] != 9999]
    Pdata.reset_index(drop=True, inplace=True)

    columns_to_Std_normalize = ['C/N0', 'pseudorange', 'Pseudorange_residual', 'Pr_rate_consitency'
                              ,'sat_clock_error']
    columns_to_M_normalize = ['Elevation', 'Azimuth', 'Sat_pos_x', 'Sat_pos_y', 'Sat_pos_z']
    std_std = StandardScaler()
    std_m = MinMaxScaler()
    Pdata[columns_to_Std_normalize] = std_std.fit_transform(Pdata[columns_to_Std_normalize])
    Pdata[columns_to_M_normalize] = std_m.fit_transform(Pdata[columns_to_M_normalize])
    return Pdata

def get_grouplist(dataframe, grouplabel, sortlabel):
    groups = dataframe.groupby(grouplabel)#groupby返回一个group对象，1269
    group_list = []
    for grouplabel, group in groups:  # 每一个group是一个tuple,(sortlabel,group),group仍是df结构，也可以用group[0],group[1]选择
        group = group.sort_values(by=sortlabel)
        group = group.loc[:,['C/N0', 'Elevation', 'Azimuth', 'pseudorange','Pseudorange_residual', 'Normalized_Pseudorange_residual',
                             'Pr_rate_consitency', 'Sat_pos_x', 'Sat_pos_y', 'Sat_pos_z',
                            'sat_clock_error', 'LOS/NLOS_label']].values
        group_list.append(group) # ndarray 10+1

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
        return len(self.data1)  # len(self.data1) = len(self.data2)


def data_split(dataset,train_ratio):
    # 计算分割的长度

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

if __name__ == "__main__":
    torch.manual_seed(123)

    P1 = openpickle('P1.pickle')
    P2 = openpickle('P2.pickle')
    P3 = openpickle('P3.pickle')
    P4 = openpickle('P4.pickle')
    P5 = openpickle('P5.pickle')
    P6 = openpickle('P6.pickle')
    P7 = openpickle('P7.pickle')

    A = pd.concat([P1, P2, P3])
    B = pd.concat([P4, P5])
    C = P6
    D = P7

    A = get_grouplist(A,'GPS_Time(s)','PRN')
    B = get_grouplist(B,'GPS_Time(s)','PRN')
    C = get_grouplist(C,'GPS_Time(s)','PRN')
    D = get_grouplist(D,'GPS_Time(s)','PRN')

    A_dataset = createData(A)
    B_dataset = createData(B)
    C_dataset = createData(C)
    D_dataset = createData(D)

    A_train, A_test = data_split(A_dataset, 0.7)
    B_train, B_test = data_split(B_dataset, 0.7)
    C_train, C_test = data_split(C_dataset, 0.7)
    D_train, D_test = data_split(D_dataset, 0.7)


    with open('dataset//A_train.pickle', 'wb') as file:
        pickle.dump(A_train, file)
    with open('dataset//A_test.pickle', 'wb') as file:
        pickle.dump(A_test, file)

    with open('dataset//B_train.pickle', 'wb') as file:
        pickle.dump(B_train, file)
    with open('dataset//B_test.pickle', 'wb') as file:
        pickle.dump(B_test, file)

    with open('dataset//C_train.pickle', 'wb') as file:
        pickle.dump(C_train, file)
    with open('dataset//C_test.pickle', 'wb') as file:
        pickle.dump(C_test, file)

    with open('dataset//D_train.pickle', 'wb') as file:
        pickle.dump(D_train, file)
    with open('dataset//D_test.pickle', 'wb') as file:
        pickle.dump(D_test, file)

