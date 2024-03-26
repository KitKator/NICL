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

        # 注册 forward hook 函数
        self.forward_hook_handle = self.fc_after_concat1.register_forward_hook(self.forward_hook)

        # 注册 backward hook 函数
        self.backward_hook_handle = self.fc_after_concat1.register_backward_hook(self.backward_hook)

        # 用于存储 forward 和 backward hook 记录的输入和输入的梯度
        self.forward_input = None
        self.backward_input_gradient = None

    def forward_hook(self, module, input, output):
        # 在正向传播时记录 fc_after_concat1 层的输入
        self.forward_input = input[0]

    def backward_hook(self, module, grad_input, grad_output):
        # 在反向传播时记录 fc_after_concat1 层的输入的梯度
        self.backward_input_gradient = grad_input[1]

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

        # Fully Connected layers after concatenation
        con_out = self.fc_after_concat1(concatenated_output)
        con_out = self.relu(con_out)
        con_out = self.fc_after_concat2(con_out)
        con_out = self.relu(con_out)

        output = self.fc_final(con_out)

        return output

    def init_weights_he(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def remove_hooks(self):
        # 移除 forward 和 backward hook 函数
        self.forward_hook_handle.remove()
        self.backward_hook_handle.remove()

# # 初始化模型
# model = model(input_dim=..., fc_outdim=..., lstm_outdim=...)
#
# # 正向传播
# input1 = ...
# input2 = ...
# output = model(input1, input2)
#
# # 获取在正向传播时记录的 fc_after_concat1 层输入
# forward_input = model.forward_input
#
# # 反向传播
# loss = ...
# loss.backward()
#
# # 获取在反向传播时记录的 fc_after_concat1 层输入的梯度
# backward_input_gradient = model.backward_input_gradient
#
# # 移除 forward 和 backward hook 函数
# model.remove_hooks()
