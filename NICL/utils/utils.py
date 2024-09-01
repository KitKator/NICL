
import torch

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader

def myDataLoader(dataset, batch_size):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=lambda batch: (
        pad_sequence([item[0] for item in batch], batch_first=True),  # input1
        pack_padded_sequence(pad_sequence([item[1] for item in batch], batch_first=True), lengths=[item[1].size(0) for item in batch], batch_first=True, enforce_sorted=False),  # input2
        torch.stack([item[2] for item in batch])))  # labels
    return dataloader


def get_correct(predicted_labels,true_labels):
    predicted_classes = torch.argmax(predicted_labels, dim=1)
    true_classes = torch.argmax(true_labels, dim=1)
    correct_predictions = torch.eq(predicted_classes, true_classes)
    correct = torch.sum(correct_predictions).item()
    return correct

def print_confusion_matrix(y_true, y_pred):
    y_true_counts = Counter(y_true)
    y_pred_counts = Counter(y_pred)
    cm = confusion_matrix(y_true, y_pred)

