import torch
import torch.nn.functional as F



def EBKD_loss(teacher_alpha, teacher_Av, student_alpha, student_Av):
    # 计算 Qv

    # teacher_Qv = F.relu(teacher_alpha * teacher_Av)
    # student_Qv = F.relu(student_alpha * student_Av)
    teacher_Qv = teacher_alpha * teacher_Av
    student_Qv = student_alpha * student_Av

    temp_teacher_alpha = teacher_alpha.detach().cpu().numpy()
    temp_teacher_Av = teacher_Av.detach().cpu().numpy()
    temp_student_alpha = student_alpha.detach().cpu().numpy()
    temp_student_Av = student_Av.detach().cpu().numpy()

    temp_teacher_Qv = teacher_Qv.detach().cpu().numpy()
    temp_student_Qv = student_Qv.detach().cpu().numpy()

    teacher_Qv_norm = F.normalize(teacher_Qv, p=2, dim=1)
    student_Qv_norm = F.normalize(student_Qv, p=2, dim=1)

    temp_teacher_Qv_norm = teacher_Qv_norm.detach().cpu().numpy()
    temp_student_Qv_norm = student_Qv_norm.detach().cpu().numpy()

    # distances = torch.norm(teacher_Qv_norm - student_Qv_norm, p=2, dim=1, keepdim=True)
    distances = torch.norm(teacher_Qv - student_Qv, p=2, dim=1, keepdim=True)

    loss = torch.mean(distances)
    
    return loss