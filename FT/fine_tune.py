import os
import pickle

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from data.getdataset import MyDataset
from utils.utils import myDataLoader, get_correct, makedir, pltshow1
from train.train import train, test
from model.model import model, init_weights_he

dataset_root = ''

with open(os.path.join(dataset_root, 'A_train.pickle'), 'rb') as file:
    A_train = pickle.load(file)
with open(os.path.join(dataset_root, 'A_test.pickle'), 'rb') as file:
    A_test = pickle.load(file)
with open(os.path.join(dataset_root, 'B_train.pickle'), 'rb') as file:
    B_train = pickle.load(file)
with open(os.path.join(dataset_root, 'B_test.pickle'), 'rb') as file:
    B_test = pickle.load(file)
with open(os.path.join(dataset_root, 'C_train.pickle'), 'rb') as file:
    C_train = pickle.load(file)
with open(os.path.join(dataset_root, 'C_test.pickle'), 'rb') as file:
    C_test = pickle.load(file)

ROOT = ''
LR = 0.001
BATCH_SIZE = 64
N_EPOCHS = 50
info = ''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random_seed = 123
torch.manual_seed(123)

input_dim = 11
fc_outdim = 64
lstm_outdim = 360
loss_list = []
acc_list = []
acc_list_1 = []
acc_list_2 = []
acc_list_3 = []
loss_list_test_1 = []
loss_list_test_2 = []
loss_list_test_3 = []
RESULT_TRAIN = []
RESULT_TEST1 = []
RESULT_TEST2 = []
RESULT_TEST3 = []

A_train = myDataLoader(A_train, BATCH_SIZE)
A_test = myDataLoader(A_test, BATCH_SIZE)
B_train = myDataLoader(B_train, BATCH_SIZE)
B_test = myDataLoader(B_test, BATCH_SIZE)
C_train = myDataLoader(C_train, BATCH_SIZE)
C_test = myDataLoader(C_test, BATCH_SIZE)

if __name__ == "__main__":
    torch.cuda.empty_cache()

    model_root = ''
    model_name = ''

    model = model(input_dim, fc_outdim, lstm_outdim).to(device)
    model.load_state_dict(torch.load(os.path.join(model_root, model_name)))  # 加载模型参数

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), LR)

    train_loader = C_train
    test_loader1 = A_test
    test_loader2 = B_test
    test_loader3 = C_test

    folder1_path, folder2_path = makedir(ROOT, LR, BATCH_SIZE, N_EPOCHS, info)

    for e in tqdm(range(0, N_EPOCHS+1)):

        if(e == 0):
            torch.save(model.state_dict(), os.path.join(folder2_path, 'epoch_{}_model.pth'.format(e)))
            test(model, criterion, test_loader1, e, N_EPOCHS, device, RESULT_TEST1, loss_list_test_1, acc_list_1)
            test(model, criterion, test_loader2, e, N_EPOCHS, device, RESULT_TEST2, loss_list_test_2, acc_list_2)
            test(model, criterion, test_loader3, e, N_EPOCHS, device, RESULT_TEST3, loss_list_test_3, acc_list_3)
        else:
            model = train(model, criterion, optimizer, train_loader, e,
                          N_EPOCHS, device, RESULT_TRAIN, loss_list, acc_list)

            torch.save(model.state_dict(), os.path.join(folder2_path, 'epoch_{}_model.pth'.format(e)))

            test(model, criterion, test_loader1, e, N_EPOCHS, device, RESULT_TEST1, loss_list_test_1, acc_list_1)
            test(model, criterion, test_loader2, e, N_EPOCHS, device, RESULT_TEST2, loss_list_test_2, acc_list_2)
            test(model, criterion, test_loader3, e, N_EPOCHS, device, RESULT_TEST3, loss_list_test_3, acc_list_3)

    res_train = np.asarray(RESULT_TRAIN)
    res_test1 = np.asarray(RESULT_TEST1)
    res_test2 = np.asarray(RESULT_TEST2)
    res_test3 = np.asarray(RESULT_TEST3)
    np.savetxt(os.path.join(folder1_path, 'TRAIN_DATA.txt'), res_train, fmt='%.6f', delimiter=',')
    np.savetxt(os.path.join(folder1_path, 'TEST_DATA1.txt'), res_test1, fmt='%.6f', delimiter=',')
    np.savetxt(os.path.join(folder1_path, 'TEST_DATA2.txt'), res_test2, fmt='%.6f', delimiter=',')
    np.savetxt(os.path.join(folder1_path, 'TEST_DATA3.txt'), res_test3, fmt='%.6f', delimiter=',')
    pltshow1(loss_list, LR, BATCH_SIZE, N_EPOCHS, folder1_path)


