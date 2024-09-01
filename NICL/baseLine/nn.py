import os
import torch
from torch import nn, optim
from tqdm import tqdm
from train.train import train, test


if __name__ == "__main__":
    torch.cuda.empty_cache()

    model.load_state_dict(torch.load(os.path.join(model_root, model_name)))  # 加载模型参数

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), LR)

    for e in tqdm(range(0, N_EPOCHS+1)):

        if(e == 0):
            torch.save(model.state_dict(), os.path.join(folder2_path, 'epoch_{}_model.pth'.format(e)))
        else:
            model = train(model, criterion, optimizer, train_loader, e,
                          N_EPOCHS, device, RESULT_TRAIN, loss_list, acc_list)

            torch.save(model.state_dict(), os.path.join(folder2_path, 'epoch_{}_model.pth'.format(e)))



