import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence
from tqdm import tqdm
from utils.utils import get_correct


def ebkd_train(teacher_model, stu_model, criterion1, criterion2, alpha,
             optimizer, dataloader, epoch, N_EPOCHS, device,
          RESULT_TRAIN, loss_list, acc_list):
    loss_epoch = 0.
    correct = 0.

    # Setup model
    stu_model.train()

    for batch_idx, (input1, input2, labels) in enumerate(dataloader):

        optimizer.zero_grad()

        if device == 'cuda':
            input1 = input1.to(device)
            input2 = input2.to(device)
            labels = labels.to(device).view(-1, 1)
            one_hot_labels = torch.nn.functional.one_hot(labels.long(), num_classes=2).squeeze(1).float()

        # Train model using source data
        stu_output = stu_model(input1, input2)
        teacher_output = teacher_model(input1, input2)

        stu_index = torch.argmax(stu_output, dim=1)
        stu_target =stu_output[range(stu_output.size(0)), stu_index]
        log_stu_target = torch.sigmoid(stu_target).mean()
        log_stu_target.backward(retain_graph=True)


        teacher_index = torch.argmax(teacher_output, dim=1)
        teacher_target = teacher_output[range(teacher_output.size(0)), teacher_index]
        log_teacher_target = torch.sigmoid(teacher_target).mean()
        log_teacher_target.backward(retain_graph=True)


        teacher_backward_gradient = teacher_model.backward_input_gradient
        student_backward_gradient = stu_model.backward_input_gradient

        teacher_forward_input = teacher_model.forward_input
        student_forward_input = stu_model.forward_input

        BCE_loss = criterion1(stu_output, one_hot_labels)
        KL_loss = criterion2(stu_output, teacher_output)
        # loss = BCE_loss * (1 - alpha) + KL_loss * alpha

        ebkd_loss = EBKD_loss(teacher_forward_input, teacher_backward_gradient,
                              student_forward_input, student_backward_gradient)

        loss = BCE_loss * (1/2) + KL_loss  + ebkd_loss
        loss.backward()

        optimizer.step()


        correct += get_correct(stu_output, one_hot_labels)
        loss_epoch += loss.item()


    loss_epoch /= len(dataloader)
    acc = correct * 100. / len(dataloader.dataset)

    loss_list.append(loss_epoch)
    acc_list.append(acc)

    res_e = 'Epoch: [{}/{}], loss: {:.6f}, correct: [{}/{}], training accuracy: {:.4f}%'.format(
        epoch, N_EPOCHS, loss_epoch, correct, len(dataloader.dataset), acc)
    tqdm.write(res_e)

    RESULT_TRAIN.append([epoch, loss_epoch, correct, acc])

    return stu_model


def EBKD_loss(teacher_alpha, teacher_Av, student_alpha, student_Av):
    teacher_Qv = F.relu(teacher_alpha * teacher_Av)
    student_Qv = F.relu(student_alpha * student_Av)

    teacher_Qv_norm = F.normalize(teacher_Qv, p=2, dim=1)
    student_Qv_norm = F.normalize(student_Qv, p=2, dim=1)

    distances = torch.norm(teacher_Qv_norm - student_Qv_norm, p=2, dim=1, keepdim=True)

    loss = torch.mean(distances)

    return loss

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='mean') * (self.T**2)
        return loss