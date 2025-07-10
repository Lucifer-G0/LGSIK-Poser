"""
    loss definition
"""
import torch
from torch import nn


class PoseJointLoss(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
        self.parents = torch.Tensor(parents).long().to(device)
        self.device = device

        self.sparse_indexes = [7, 8, 15, 20, 21]
        self.indices_no_sparse = []

        for i in range(0, 22):
            if i not in self.sparse_indexes:
                self.indices_no_sparse.append(i)
        self.indices_no_sparse = torch.tensor(self.indices_no_sparse).to(self.device)

    def forward(self,
                pred_local_rotation,
                pred_joint_position,
                pred_shape,
                gt_local_rotation,
                gt_joint_position,
                gt_shape
                ):
        assert pred_local_rotation.shape == gt_local_rotation.shape and \
               pred_joint_position.shape == gt_joint_position.shape and \
               pred_shape.shape == gt_shape.shape, "GT and Pred have different shape"

        """
                :return: A tuple containing:
                 - pred_pose: 预测关节局部旋转(batch_size, frames, 22 * 6)
                 - pred_global_pose: 预测关节全局旋转，6D表示法 (batch_size, frames, 22 * 6)
                 - pred_joint_position: 预测关节位置（无根位移未对齐） (batch_size, frames, 22, 3)
                """
        batch_size = pred_local_rotation.shape[0]
        frames = pred_local_rotation.shape[1]
        joint_poses = pred_joint_position.reshape(batch_size, frames, 22, 3)
        target_joint_poses = gt_joint_position.reshape(batch_size, frames, 22, 3)
        joint_rot_mat = pred_local_rotation.reshape(batch_size, frames, 22, 6)
        target_joint_rot_mat = gt_local_rotation.reshape(batch_size, frames, 22, 6)
        shape_loss = self.mse(pred_shape, pred_shape.mean(dim=1, keepdim=True).repeat(1, pred_shape.shape[1], 1))

        # positions error
        loss_ee = self.mse(joint_poses[:, :, self.sparse_indexes, :],
                           target_joint_poses[:, :, self.sparse_indexes, :])

        loss_root = self.mse(joint_poses[:, :, [0], :], target_joint_poses[:, :, [0], :])
        loss_root += self.mse(joint_rot_mat[:, :, [0], :], target_joint_rot_mat[:, :, [0], :])

        # regularization
        loss_ee_reg = self.mse(target_joint_poses[:, :, self.indices_no_sparse, :],
                               joint_poses[:, :, self.indices_no_sparse, :])
        loss_ee_reg += self.mse(target_joint_rot_mat[:, :, self.indices_no_sparse, :],
                                joint_rot_mat[:, :, self.indices_no_sparse, :])

        # 计算Vel(1)，即相邻1帧间的速度
        vel_error_1 = torch.norm(
            target_joint_poses[:, 1:] - target_joint_poses[:, :-1] - (
                    joint_poses[:, 1:] - joint_poses[:, :-1]),
            dim=-1)

        # 计算Vel(3)，即相邻3帧间的速度
        vel_error_3 = torch.norm(
            target_joint_poses[:, 3:] - target_joint_poses[:, :- 3] - (
                    joint_poses[:, 3:] - joint_poses[:, :- 3]),
            dim=-1)

        # 计算Vel(5)，即相邻5帧间的速度
        vel_error_5 = torch.norm(
            target_joint_poses[:, 5:] - target_joint_poses[:, :- 5] - (
                    joint_poses[:, 5:] - joint_poses[:, :- 5]),
            dim=-1)

        # 将Vel(1), Vel(3), Vel(5)相加
        loss_vel = torch.mean(vel_error_1) + torch.mean(vel_error_3) + torch.mean(vel_error_5)

        # return loss_ee * self.param["lambda_ee"] + 10 * loss_vel
        return loss_root * 100, loss_ee * 100, loss_ee_reg * 10, loss_vel,shape_loss,shape_loss+ loss_root * 100 + loss_ee * 100 + loss_ee_reg * 10 + loss_vel
