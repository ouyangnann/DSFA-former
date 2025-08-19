import torch
import torch.nn.functional as F

def kd_normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def dkd_loss(logits_student_in, logits_teacher_in, target, alpha, beta, temperature, logit_stand):
    logits_student = kd_normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = kd_normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    # 使用基于熵的动态掩码
    gt_mask = dynamic_entropy_mask(logits_student, target, entropy_threshold=1.0)
    other_mask = ~gt_mask  # 其他掩码是相反的

    pred_student = F.softmax(logits_student / temperature, dim=-1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=-1)
    
    # 应用掩码
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)

    log_pred_student = torch.log(pred_student + 1e-7)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
        * (temperature**2)
    )
    
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=-1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=-1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean')
        * (temperature**2)
    )
    
    return alpha * tckd_loss + beta * nckd_loss


def compute_entropy(target):
    """
    计算目标标签的熵，表示每个时间步的标签不确定性。
    """
 
    batch_size, seq_len, num_nodes = target.shape
    target_prob = F.softmax(target.float(), dim=-1)  # 将 target 转化为概率分布
    entropy = -torch.sum(target_prob * torch.log(target_prob + 1e-7), dim=-1)  # 计算熵

    return entropy

def dynamic_entropy_mask(logits, target, entropy_threshold=1.0):
    """
    根据目标标签的熵值生成动态掩码，熵值大于阈值的时间步会被标记为目标时刻。
    """
    entropy = compute_entropy(target)  # 计算每个时间步的熵
    mask = entropy > entropy_threshold  # 熵值大于阈值的位置视为目标时刻
    mask = mask.unsqueeze(-1).expand_as(logits) 
    return mask  # 维度为 (batch_size, seq_len, num_nodes)

def _get_gt_mask(logits, target):
    """
    生成用于区分目标位置的掩码。
    """
    batch_size, seq_len, num_nodes = logits.shape
    target = target.reshape(batch_size, seq_len, num_nodes).long()

    # 确保目标值在 [0, num_nodes) 范围内
    target = target.clamp(0, num_nodes - 1)

    mask = torch.zeros_like(logits).scatter_(2, target, 1).bool()
    return mask  # 维度为 (batch_size, seq_len, num_nodes)

def _get_other_mask(logits, target):
    """
    生成用于区分非目标位置的掩码。
    """
    batch_size, seq_len, num_nodes = logits.shape
    target = target.reshape(batch_size, seq_len, num_nodes).long()

    # 确保目标值在 [0, num_nodes) 范围内
    target = target.clamp(0, num_nodes - 1)

    mask = torch.ones_like(logits).scatter_(2, target, 0).bool()
    return mask  # 维度为 (batch_size, seq_len, num_nodes)

def cat_mask(t, mask1, mask2):
    """
    根据给定的掩码，将预测结果（t）与掩码进行加权。
    """
    t1 = (t * mask1).sum(dim=2, keepdims=True)  # 按节点维度求和
    t2 = (t * mask2).sum(dim=2, keepdims=True)  # 按节点维度求和
    rt = torch.cat([t1, t2], dim=2)  # 按节点维度拼接结果
    return rt  # 返回合并后的结果，维度为 (batch_size, seq_len, 2)

def distillation_loss(student_outputs, teacher_outputs, 
                        true_labels, temperature=3.0, alpha=1.0, beta=1.0, logit_stand=True):
    #print('input',student_outputs.max(),teacher_outputs.max(),student_en_ou.max(), teacher_en_ou.max())
    
    loss_dkd = dkd_loss(student_outputs, teacher_outputs, true_labels, alpha, beta, temperature, logit_stand)

    return  loss_dkd


def compute_sequence_similarity(x, r):
    """
    计算序列之间的相似性。
    :param x: 输入特征张量 [batch_size, seq_len, num_node]
    :param r: 相似性阈值
    :return: 相似性矩阵 [batch_size]
    """
    batch_size, seq_len, num_node = x.shape
    similarity = torch.zeros(batch_size)

    diff = torch.abs(x - x.mean(dim=1, keepdim=True))  # 计算每个样本与其均值的差异
 
    # 求每个节点的最大差异，得到一个 (batch_size, n_samples) 的矩阵
    max_diff = torch.max(diff, dim=-1).values  # 计算每个样本的最大差异值
    #print(max_diff)
    # 如果差异小于阈值 r，认为它们相似，得到一个布尔值矩阵
    similarity = (max_diff < r).float().mean(dim=-1)  # 计算每个样本的相似性均值

    return similarity

def compute_sequence_complexity(similarity):
    """
    计算序列的复杂度（log of similarity).
    :param similarity: [batch_size] 的相似性矩阵
    :return: 每个样本的复杂度 [batch_size]
    """
    complexity = torch.log(1 + similarity)  # 使用对数来衡量复杂度
    return complexity

def PaENLoss(original_features, generated_features, r=0.1):
    """
    计算 Patch Entropy Loss，确保生成特征有更高的信息复杂度。
    :param original_features: 原始特征 [batch_size, seq_len, num_node]
    :param generated_features: 生成特征 [batch_size, seq_len, num_node]
    :param r: 相似性阈值
    :return: Patch Entropy Loss
    """
    min_val = torch.min(original_features) # [batch_size, 1, num_node]
    max_val = torch.max(original_features)  # [batch_size, 1, num_node]

    original_features = (original_features - min_val) / (max_val - min_val + 1e-7)  # 加上一个小的常数避免除以零
    generated_features = (generated_features - min_val) / (max_val - min_val + 1e-7)  # 加上一个小的常数避免除以零
    # 计算原始特征的相似性和复杂度
    similarity_LF = compute_sequence_similarity(original_features, r)  # [batch_size]
    complexity_LF = compute_sequence_complexity(similarity_LF)  # [batch_size]

    # 计算生成特征的相似性和复杂度
    similarity_LE = compute_sequence_similarity(generated_features, r)  # [batch_size]
    complexity_LE = compute_sequence_complexity(similarity_LE)  # [batch_size]

    # 计算熵增：最大化生成特征的复杂度
    # 计算原始特征和生成特征的复杂度差异
    PaEn = -torch.mean(complexity_LE - complexity_LF)  # [batch_size]
    return PaEn

"""
def calculate_dtw(A, B):
  
    seq_len, num_node = A.shape
    # 计算欧氏距离矩阵 (A和B之间的差异)
    dist_matrix = torch.abs(A.unsqueeze(1) - B.unsqueeze(0))  # [seq_len, seq_len, num_node]
    
    # 初始化 DTW 矩阵，存储累计的最小路径
    D = dist_matrix.clone()  # 不修改原 dist_matrix

    # 使用广播初始化第一行和第一列
    D[0, 0, :] = dist_matrix[0, 0, :]
    D[1:, 0, :] = dist_matrix[1:, 0, :] + D[:-1, 0, :]
    D[0, 1:, :] = dist_matrix[0, 1:, :] + D[0, :-1, :]

    # 填充剩余的矩阵（通过广播方式避免显式的 for 循环）
    for i in range(1, seq_len):
        D[i, 1:, :] = dist_matrix[i, 1:, :] + torch.min(
            torch.stack([D[i-1, 1:, :], D[i, :-1, :], D[i-1, :-1, :]], dim=0),
            dim=0
        ).values

    # 计算最终的 DTW 误差，矩阵的右下角的值
    dtw_loss = D[-1, -1, :]  # 取右下角的每个节点的DTW距离
    dtw_loss = dtw_loss.mean()  # 对所有节点的DTW距离取平均值
    
    return dtw_loss


def batch_calculate_dtw(A, B):

    batch_size = A.shape[0]
    dtw_losses = []  # 用列表存储每个样本的DTW损失

    for i in range(batch_size):
        dtw_losses.append(calculate_dtw(A[i], B[i]))  # 使用append来存储每个样本的DTW损失
    
    # 转换为tensor并计算批量的平均损失
    dtw_loss_batch = torch.tensor(dtw_losses, device=A.device).mean()
    
    return dtw_loss_batch


def cosine_similarity_error(x1, x2):
    # 计算余弦相似度
    cos_sim = F.cosine_similarity(x1, x2, dim=-1)  # [batch_size, seq_len]
    
    # 计算误差：1 - cosine_similarity
    error = 1 - cos_sim
    return error.mean(dim=-1).mean()
"""


"""
def nt_xent_loss(z_i, z_j, temperature=0.5):


    # 对特征维度 (f_dim) 进行聚合，计算每个时间序列样本在所有特征上的平均表示
    # z_i 和 z_j 最初的形状是 (batch_size, seq_len, f_dim)
    # 经过聚合后，z_i 和 z_j 的形状变为 (batch_size, seq_len)，每个样本的特征信息被压缩为单一值
    z_i = z_i.mean(dim=-1)  # 对最后一个维度 (f_dim) 进行求平均操作
    z_j = z_j.mean(dim=-1)  # 对最后一个维度 (f_dim) 进行求平均操作
    
    # 对每个时间步的特征向量归一化，使得它们具有单位范数，便于计算余弦相似度
    z_i = F.normalize(z_i, dim=-1)
    z_j = F.normalize(z_j, dim=-1)

    # 将 z_i 和 z_j 在样本维度上拼接起来，形成一个双倍 batch 大小的表示矩阵
    representations = torch.cat([z_i, z_j], dim=0)
    
    # 计算样本之间的余弦相似度矩阵，大小为 (2 * batch_size, 2 * batch_size)
    similarity_matrix = torch.matmul(representations, representations.T)

    # 构建标签，用于标记正样本对
    batch_size = z_i.shape[0]
    labels = torch.arange(batch_size, device=z_i.device)
    labels = torch.cat([labels, labels], dim=0)

    # 创建掩码，排除对角线上的自身匹配
    mask = torch.eye(batch_size * 2, device=z_i.device).bool()
    similarity_matrix = similarity_matrix[~mask].view(batch_size * 2, -1)

    # 计算正样本对的相似度
    positives = torch.exp(similarity_matrix[range(batch_size * 2), labels] / temperature)
    
    # 计算分母部分，包含所有样本对的相似度总和
    denominator = torch.sum(torch.exp(similarity_matrix / temperature), dim=1)

    # 计算对比损失：期望正样本对的相似度尽可能高，负样本对的相似度尽可能低
    loss = -torch.log(positives / denominator).mean()

    return loss
"""
"""
def contrastive_loss(x1, x2, tau=0.01):
    batch_size, seq_len, feature_dim = x1.shape
    x1_flat = x1.reshape(batch_size, seq_len * feature_dim)
    x2_flat = x2.reshape(batch_size, seq_len * feature_dim)

    # 对输入进行归一化
    x1_norm = F.normalize(x1_flat, dim=-1)
    x2_norm = F.normalize(x2_flat, dim=-1)

    # 计算正样本的相似度（x1 和 x2 对应位置的余弦相似度）
    l_pos = torch.sum(x1_norm * x2_norm, dim=-1, keepdim=True)  # [N, 1]

    # 计算负样本的相似度（x1 与 x2 中所有其他样本的余弦相似度）
    l_neg = torch.matmul(x1_norm, x2_norm.T)  # [N, N]

    # 创建掩码，排除正样本的位置
    mask = torch.eye(batch_size, dtype=torch.bool).to(x1.device)
    l_neg.masked_fill_(mask, float('-inf'))

    # 将正样本和负样本的相似度合并
    logits = torch.cat([l_pos / tau, l_neg / tau], dim=1)  # [N, N+1]

    # 标签：正样本的索引为 0
    labels = torch.zeros(batch_size, dtype=torch.long).to(x1.device)
    # 计算对比损失
    loss = F.cross_entropy(logits, labels)

    return loss                 

"""