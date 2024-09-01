import copy
import os

if __name__ == "__main__":
    torch.cuda.empty_cache()

    teacher_model = model(input_dim, fc_outdim, lstm_outdim).to(device)
    teacher_model.load_state_dict(torch.load(os.path.join(model_root, model_name)))  # 加载模型参数
    stu_model = copy.deepcopy(teacher_model)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = DistillKL(T)
    optimizer = optim.Adam(stu_model.parameters(), LR)

    folder1_path, folder2_path = makedir(ROOT, LR, BATCH_SIZE, N_EPOCHS, info)

    stu_model.lstm.flatten_parameters()
    teacher_model.lstm.flatten_parameters()

    for e in tqdm(range(0, N_EPOCHS+1)):
        if (e == 0):
            torch.save(stu_model.state_dict(), os.path.join(folder2_path, 'epoch_{}_stu_model.pth'.format(e)))
        else:
            stu_model = ebkd_train(teacher_model, stu_model, criterion1, criterion2, alpha,
             optimizer, train_loader, e, N_EPOCHS, device, RESULT_TRAIN, loss_list, acc_list)

            torch.save(stu_model.state_dict(), os.path.join(folder2_path, 'epoch_{}_stu_model.pth'.format(e)))


