import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence

class model(nn.Module):
    def __init__(self, input_dim, fc_outdim, lstm_outdim):
        super(model, self).__init__()

        # Fully Connected layers for input1
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, fc_outdim)

        # LSTM layer for input2
        self.lstm = nn.LSTM(input_dim, lstm_outdim, batch_first=True)

        # Fully Connected layers after concatenation
        self.fc_after_concat1 = nn.Linear(fc_outdim + lstm_outdim, 64)
        self.fc_after_concat2 = nn.Linear(64, 64)
        self.fc_final = nn.Linear(64, 2)

        self.relu = nn.ReLU()

    def forward(self, input1, input2):
        # Processing input1
        fc_out = self.fc1(input1)
        fc_out = self.relu(fc_out)
        fc_out = self.fc2(fc_out)
        fc_out = self.relu(fc_out)

        lstm_out, _ = self.lstm(input2)

        # 解压缩 PackedSequence，仅考虑有效长度
        unpacked_sequence, lengths_unpacked = pad_packed_sequence(lstm_out, batch_first=True,
                                                                  padding_value=float('nan'))

        # 获取最后一个时间步的输出，仅考虑有效长度
        last_time_step_output = unpacked_sequence[torch.arange(unpacked_sequence.size(0)), lengths_unpacked - 1, :]

        # Concatenating processed input1 and the entire output sequence from LSTM
        concatenated_output = torch.cat((fc_out, last_time_step_output), dim=1)

        temp1 = concatenated_output.detach().cpu().numpy()

        # Fully Connected layers after concatenation
        con_out = self.fc_after_concat1(concatenated_output)
        temp2 = con_out.detach().cpu().numpy()
        con_out = self.relu(con_out)
        con_out = self.fc_after_concat2(con_out)
        temp3 = con_out.detach().cpu().numpy()

        con_out = self.relu(con_out)

        output = self.fc_final(con_out)

        return output

def init_weights_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)



# Leaky
# PReLU
# ELU
