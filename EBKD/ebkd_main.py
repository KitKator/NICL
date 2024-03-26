import copy
import os
import pickle
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from IL.KLloss import DistillKL
from data.getdataset import MyDataset
from utils.utils import myDataLoader, get_correct, makedir, pltshow1
from train.train import test
from EBKD.model_backup import model
from EBKD.ebkd_train import ebkd_train

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
N_EPOCHS = 30
info = 'test2'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random_seed = 123
torch.manual_seed(123)

input_dim = 11
fc_outdim = 64
lstm_outdim = 360
T = 1
alpha = 0.5
loss_list = []
acc_list = []
acc_list_test1 = []
acc_list_test2 = []
acc_list_test3 = []
loss_list_test1 = []
loss_list_test2 = []
loss_list_test3 = []
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

    model_root = 'E:\\Experiments\\ABCD\\ebkd0\\B\\Train_LR0.001_batch64_epoch30_adam_test\\model_epoch_ir=0.001'
    model_name = 'epoch_28_stu_model.pth'

    teacher_model = model(input_dim, fc_outdim, lstm_outdim).to(device)
    teacher_model.load_state_dict(torch.load(os.path.join(model_root, model_name)))  # 加载模型参数
    stu_model = copy.deepcopy(teacher_model)


    criterion1 = nn.CrossEntropyLoss()
    criterion2 = DistillKL(T)
    optimizer = optim.Adam(stu_model.parameters(), LR)

    train_loader = C_train
    test_loader1 = A_test
    test_loader2 = B_test
    test_loader3 = C_test

    folder1_path, folder2_path = makedir(ROOT, LR, BATCH_SIZE, N_EPOCHS, info)

    stu_model.lstm.flatten_parameters()
    teacher_model.lstm.flatten_parameters()

    for e in tqdm(range(0, N_EPOCHS+1)):
        if (e == 0):
            torch.save(stu_model.state_dict(), os.path.join(folder2_path, 'epoch_{}_stu_model.pth'.format(e)))
            test(stu_model, criterion1, test_loader1, e, N_EPOCHS, device, RESULT_TEST1, loss_list_test1,
                 acc_list_test1)
            test(stu_model, criterion2, test_loader2, e, N_EPOCHS, device, RESULT_TEST2, loss_list_test2,
                 acc_list_test2)
            test(stu_model, criterion1, test_loader3, e, N_EPOCHS, device, RESULT_TEST3, loss_list_test3,
                 acc_list_test3)
        else:
            stu_model = ebkd_train(teacher_model, stu_model, criterion1, criterion2, alpha,
             optimizer, train_loader, e, N_EPOCHS, device, RESULT_TRAIN, loss_list, acc_list)

            torch.save(stu_model.state_dict(), os.path.join(folder2_path, 'epoch_{}_stu_model.pth'.format(e)))

            test(stu_model, criterion1, test_loader1, e, N_EPOCHS, device, RESULT_TEST1, loss_list_test1, acc_list_test1)
            test(stu_model, criterion2, test_loader2, e, N_EPOCHS, device, RESULT_TEST2, loss_list_test2, acc_list_test2)
            test(stu_model, criterion1, test_loader3, e, N_EPOCHS, device, RESULT_TEST3, loss_list_test3, acc_list_test3)

    res_train = np.asarray(RESULT_TRAIN)
    res_test1 = np.asarray(RESULT_TEST1)
    res_test2 = np.asarray(RESULT_TEST2)
    res_test3 = np.asarray(RESULT_TEST3)
    np.savetxt(os.path.join(folder1_path, 'TRAIN_DATA.txt'), res_train, fmt='%.6f', delimiter=',')
    np.savetxt(os.path.join(folder1_path, 'TEST_DATA.txt'), res_test1, fmt='%.6f', delimiter=',')
    np.savetxt(os.path.join(folder1_path, 'TEST_DATA2.txt'), res_test2, fmt='%.6f', delimiter=',')
    np.savetxt(os.path.join(folder1_path, 'TEST_DATA3.txt'), res_test3, fmt='%.6f', delimiter=',')
    pltshow1(loss_list, LR, BATCH_SIZE, N_EPOCHS, folder1_path)


