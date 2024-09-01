import torch
from sklearn.metrics import confusion_matrix
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model(input_dim, fc_outdim, lstm_outdim).to(device)
model.load_state_dict(torch.load(os.path.join(model_root, model_name)))  # 加载模型参数

def test(model, dataloader, device):
    y_true = []
    y_pred = []
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for batch_idx, (input1, input2, labels) in enumerate(dataloader):
            if device == 'cuda':
                input1 = input1.to(device)
                input2 = input2.to(device)
                labels = labels.to(device).view(-1, 1)

            output = model(input1, input2)
            _, predicted = torch.max(output.data, 1)

            y_true.extend(labels.cpu().numpy().flatten())
            y_pred.extend(predicted.cpu().numpy().flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    return cm


