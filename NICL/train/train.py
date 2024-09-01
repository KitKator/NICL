import torch
from tqdm import tqdm
from utils.utils import get_correct


def train(model, criterion, optimizer, dataloader,
          epoch, N_EPOCHS, device,
          RESULT_TRAIN, loss_list, acc_list):
    loss_epoch = 0.
    correct = 0.

    # Setup model
    model.train()

    for batch_idx, (input1, input2, labels) in enumerate(dataloader):

        optimizer.zero_grad()

        if device == 'cuda':
            input1 = input1.to(device)
            input2 = input2.to(device)
            labels = labels.to(device).view(-1, 1)
            one_hot_labels = torch.nn.functional.one_hot(labels.long(), num_classes=2).squeeze(1).float()


        # Train model using source data
        output = model(input1, input2)
        loss = criterion(output, one_hot_labels)

        loss.backward()
        optimizer.step()

        correct += get_correct(output, one_hot_labels)
        loss_epoch += loss.item()


    loss_epoch /= len(dataloader)
    acc = correct * 100. / len(dataloader.dataset)

    loss_list.append(loss_epoch)
    acc_list.append(acc)

    res_e = 'Epoch: [{}/{}], loss: {:.6f}, correct: [{}/{}], training accuracy: {:.4f}%'.format(
        epoch, N_EPOCHS, loss_epoch, correct, len(dataloader.dataset), acc)
    tqdm.write(res_e)

    RESULT_TRAIN.append([epoch, loss_epoch, correct, acc])


    return model

def test(model, criterion, dataloader, epoch, N_EPOCHS, device,
          RESULT_TEST, loss_list, acc_list):

    correct = 0.
    loss_epoch = 0.

    for batch_idx, (input1, input2, labels) in enumerate(dataloader):

        if device == 'cuda':
            input1 = input1.to(device)
            input2 = input2.to(device)
            labels = labels.to(device).view(-1, 1)
            one_hot_labels = torch.nn.functional.one_hot(labels.long(), num_classes=2).squeeze(1).float()

        with torch.no_grad():
            # Train model using source data
            output = model(input1, input2)
            loss = criterion(output, one_hot_labels)

        correct += get_correct(output, one_hot_labels)
        loss_epoch += loss.item()

    loss_epoch /= len(dataloader)
    acc = correct * 100. / len(dataloader.dataset)

    # 记录到list
    loss_list.append(loss_epoch)
    acc_list.append(acc)

    res_e = 'Epoch: [{}/{}], loss: {:.6f}, correct: [{}/{}], test accuracy: {:.4f}%'.format(
        epoch, N_EPOCHS, loss_epoch, correct, len(dataloader.dataset), acc)
    tqdm.write(res_e)

    RESULT_TEST.append([epoch, loss_epoch, correct, acc])


