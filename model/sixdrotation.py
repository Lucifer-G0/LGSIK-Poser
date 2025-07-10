import torch


def normalize_vector(v, eps=1e-8):
    """对向量进行归一化，添加小的阈值以避免数值问题"""
    norm = torch.norm(v, dim=-1, keepdim=True)
    if torch.any(norm < eps):
        raise ValueError("输入向量的范数过小，接近零向量")
    return v / norm


def orthogonalize_vectors(v1, v2, eps=1e-8):
    """对两个向量进行正交化，添加小的阈值以避免数值问题"""
    v1_normalized = normalize_vector(v1, eps)

    # 通过施密特正交化方法对第二个向量进行正交化
    dot_product = torch.sum(v2 * v1_normalized, dim=-1, keepdim=True)
    v2_orthogonal = v2 - dot_product * v1_normalized

    # 检查 v2_orthogonal 是否接近零向量
    norm_2 = torch.norm(v2_orthogonal, dim=-1, keepdim=True)
    if torch.any(norm_2 < eps):
        raise ValueError("输入向量 v2 与 v1 几乎共线，无法正交化")

    v2_normalized = normalize_vector(v2_orthogonal, eps)

    return v1_normalized, v2_normalized


def normalize(pose_6d, eps=1e-8):
    """
    对6D表示法进行正则化
    :param pose_6d: 输入的6D表示法，形状为 (N, 6)
    :param eps: 小的阈值，用于数值稳定性
    :return: 正则化后的6D表示法，形状为 (N, 6)
    """
    v1 = pose_6d[..., :3]
    v2 = pose_6d[..., 3:6]
    v3 = pose_6d[..., 6:9]

    v1_reg, v2_reg = orthogonalize_vectors(v1, v2, eps)

    # 将正则化后的向量重新组合为6D表示法
    pose_6d_reg = torch.cat([v1_reg, v2_reg, v3], dim=-1)
    return pose_6d_reg

def normalizeraw(pose_6d, eps=1e-8):
    """
    对6D表示法进行正则化
    :param pose_6d: 输入的6D表示法，形状为 (N, 6)
    :param eps: 小的阈值，用于数值稳定性
    :return: 正则化后的6D表示法，形状为 (N, 6)
    """
    v1 = pose_6d[..., :3]
    v2 = pose_6d[..., 3:6]

    v1_reg, v2_reg = orthogonalize_vectors(v1, v2, eps)

    # 将正则化后的向量重新组合为6D表示法
    pose_6d_reg = torch.cat([v1_reg, v2_reg], dim=-1)
    return pose_6d_reg