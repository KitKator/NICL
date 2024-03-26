from torch import nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T#教师模型指导学生模型的程度，值越大，指导程度越高

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        #下面就是对两个模型的预测值，做分布分析，如果偏差越大，则kl散度算出来的值越大。
        #p_t表示教师模型的目标值
        #p_s表示学生模型的预测值
        loss = F.kl_div(p_s, p_t, reduction='mean') * (self.T**2)
        return loss



#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class AttentionLayer(nn.Module):
#     def __init__(self, hidden_size):
#         super(AttentionLayer, self).__init__()
#         self.linear = nn.Linear(hidden_size, 1)
#
#     def forward(self, x):
#         # 输入 x 的形状为 (batch_size, sequence_length, hidden_size)
#         energy = self.linear(x)
#         attention_weights = F.softmax(energy, dim=1)
#         attended = torch.sum(x * attention_weights, dim=1)
#         return attended
#
#
# class YourModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(YourModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.attention = AttentionLayer(hidden_size)
#         self.fc = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         attended = self.attention(lstm_out)
#         output = self.fc(attended)
#         return output
#
#
# def compute_ebkd_loss(teacher_outputs, student_outputs, temperature=1.0, lambda_ebkd=1.0):
#     # 计算交叉熵损失，用于模仿教师模型的预测
#     ce_loss = F.cross_entropy(student_outputs, torch.argmax(teacher_outputs, dim=-1), reduction='mean')
#
#     # 计算 KL 散度损失
#     teacher_probs = F.softmax(teacher_outputs / temperature, dim=-1)
#     student_probs = F.softmax(student_outputs / temperature, dim=-1)
#     kl_loss = F.kl_div(student_probs.log(), teacher_probs, reduction='batchmean')
#
#     # 计算 EBKD 损失
#     ebkd_loss = ce_loss + lambda_ebkd * kl_loss
#
#     return ebkd_loss
#
#
# # 示例使用
# # 假设 teacher_outputs 和 student_outputs 是模型在相应输入上的输出概率分布
# input_size = 10  # 你的输入特征维度
# hidden_size = 32  # LSTM 的隐藏层大小
# num_classes = 2  # 你的类别数
#
# teacher_model = YourModel(input_size, hidden_size, num_classes)
# student_model = YourModel(input_size, hidden_size, num_classes)
#
# # 在这里，你需要确保两个模型共享相同的参数，或者使用相同的预训练权重
#
# # 假设 x 是你的输入数据
# x = torch.randn((batch_size, sequence_length, input_size))
#
# # 计算模型输出
# teacher_outputs = teacher_model(x)
# student_outputs = student_model(x)
#
# # 计算 EBKD 损失
# ebkd_loss = compute_ebkd_loss(teacher_outputs, student_outputs)
#
# # 在你的训练循环中，将这个 ebkd_loss 损失加到总体损失中，然后反向传播更新学生模型的参数
