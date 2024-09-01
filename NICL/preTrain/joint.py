import os
import torch
from torch import nn, optim
from tqdm import tqdm
from model.model import model, init_weights_he


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    torch.cuda.empty_cache()

    model = model(input_dim, fc_outdim, lstm_outdim).to(device)
    model.apply(init_weights_he)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), LR)

    for e in tqdm(range(1, N_EPOCHS+1)):

        model = train(model, criterion, optimizer, train_loader, e,
                       N_EPOCHS, device, RESULT_TRAIN, loss_list, acc_list)
        torch.save(model.state_dict(), os.path.join(folder2_path, 'epoch_{}_model.pth'.format(e)))



