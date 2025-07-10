import torch
import os
import time
from tqdm import tqdm
import numpy as np
from utils import utils_transform
import math
from model.hmd_imu_model import HMDIMUModel

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
METERS_TO_CENTIMETERS = 100.0


def train_loop(configs, device, model, loss_func, optimizer, \
               lr_scheduler, train_loader, test_loader, LOG, train_summary_writer, out_dir):

    LOG.info("testing HMDIMUmodel...")
    # 训练开始之前对原始模型进行测试用于对比有无提升。
    one_epoch_loss_root, one_epoch_loss_ee, one_epoch_loss_ee_reg, one_epoch_loss_vel, one_epoch_total_loss = [], [], [], [], []
    position_error_, local_pose_error_ = [], []
    position_error_upper, local_pose_error_upper = [], []
    with torch.no_grad():
        for batch_idx, (input_feat,pred_local_pose, pred_betas, rotation_global_r6d, pred_joint_position, gt_local_pose, gt_positions,gt_betas) in tqdm(enumerate(test_loader)):
            if len(input_feat.shape) == 4:
                print("-----------------------------------------error inputfeat--------------------------------------------------------------------------")

            pred_joint_position_head_centered = pred_joint_position - pred_joint_position[:, :,15:16] + gt_positions[:, :, 15:16]
            gt_positions_head_centered = gt_positions

            loss_root, loss_ee, loss_ee_reg, loss_vel, total_loss = (
                loss_func(rotation_global_r6d, pred_joint_position_head_centered, pred_betas, gt_local_pose,
                          gt_positions_head_centered, gt_betas))

            pos_error_ = torch.mean(torch.sqrt(
                torch.sum(
                    torch.square(gt_positions_head_centered - pred_joint_position_head_centered), axis=-1
                )
            ))
            position_error_.append(pos_error_.item() * METERS_TO_CENTIMETERS)
            pred_local_pose_aa = utils_transform.sixd2aa(pred_local_pose.reshape(-1, 6).detach()).reshape(-1, 22 * 3)
            gt_local_pose_aa = utils_transform.sixd2aa(gt_local_pose.reshape(-1, 6).detach()).reshape(-1, 22 * 3)

            diff = gt_local_pose_aa - pred_local_pose_aa
            diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
            diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
            rot_error = torch.mean(torch.absolute(diff))
            local_pose_error_.append(rot_error.item() * RADIANS_TO_DEGREES)
            #---------------------------------------------------------------------------------------------
            upper_indexes = [0,3,6,9,12,13,14,15,16,17,18,19,20,21]
            pos_error_upper = torch.mean(torch.sqrt(
                torch.sum(
                    torch.square(gt_positions_head_centered[:,:,upper_indexes] - pred_joint_position_head_centered[:,:,upper_indexes]), axis=-1
                )
            ))
            position_error_upper.append(pos_error_upper.item() * METERS_TO_CENTIMETERS)

            pred_local_pose_aa = utils_transform.sixd2aa(pred_local_pose.reshape(-1, 6).detach()).reshape(-1, 22 ,3)[:,upper_indexes].reshape(-1,14*3)
            gt_local_pose_aa = utils_transform.sixd2aa(gt_local_pose.reshape(-1, 6).detach()).reshape(-1, 22 ,3)[:,upper_indexes].reshape(-1,14*3)

            diff = gt_local_pose_aa - pred_local_pose_aa
            diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
            diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
            rot_error_upper = torch.mean(torch.absolute(diff))
            local_pose_error_upper.append(rot_error_upper.item() * RADIANS_TO_DEGREES)

            one_epoch_loss_root.append(loss_root.item())
            one_epoch_loss_ee.append(loss_ee.item())
            one_epoch_loss_ee_reg.append(loss_ee_reg.item())
            one_epoch_loss_vel.append(loss_vel.item())
            one_epoch_total_loss.append(total_loss.item())

    one_epoch_loss_root = torch.tensor(one_epoch_loss_root).mean().item()
    one_epoch_loss_ee = torch.tensor(one_epoch_loss_ee).mean().item()
    one_epoch_loss_ee_reg = torch.tensor(one_epoch_loss_ee_reg).mean().item()
    one_epoch_loss_vel = torch.tensor(one_epoch_loss_vel).mean().item()
    one_epoch_total_loss = torch.tensor(one_epoch_total_loss).mean().item()

    position_error_ = torch.tensor(position_error_).mean().item()
    local_pose_error_ = torch.tensor(local_pose_error_).mean().item()
    position_error_upper = torch.tensor(position_error_upper).mean().item()
    local_pose_error_upper = torch.tensor(local_pose_error_upper).mean().item()

    epoch_info = {
        'type': 'eval',
        'epoch': 000,
        'loss_total': round(float(one_epoch_total_loss), 3),
        'loss_root': round(float(one_epoch_loss_root), 3),
        'loss_ee': round(float(one_epoch_loss_ee), 3),
        'loss_ee_reg': round(float(one_epoch_loss_ee_reg), 3),
        'loss_vel': round(float(one_epoch_loss_vel), 3),
    }
    LOG.info(epoch_info)
    LOG.info("HMD-Poser MPJPE {} ".format(position_error_))
    LOG.info("HMD-Poser MPJRE_(including root) {}".format(local_pose_error_))
    LOG.info("HMD-Poser Upper MPJPE {} ".format(position_error_upper))
    LOG.info("HMD-Poser Upper MPJRE_(including root) {}".format(local_pose_error_upper))

    global_step = 0
    best_train_loss, best_test_loss, best_position, best_local_pose = float("inf"), float("inf"), float("inf"), float(
        "inf")
    for epoch in range(configs.train_config.epochs):
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        LOG.info(f'Training epoch: {epoch + 1}/{configs.train_config.epochs}, LR: {current_lr:.8f}')

        batch_start = time.time()

        one_epoch_loss_root, one_epoch_loss_ee, one_epoch_loss_ee_reg, one_epoch_loss_vel, one_epoch_total_loss = [], [], [], [], []
        for batch_idx, (input_feat,pred_local_pose, pred_betas, rotation_global_r6d, pred_joint_position, gt_local_pose, gt_positions,gt_betas) in tqdm(enumerate(train_loader)):
            global_step += 1
            optimizer.zero_grad()

            # 在这个部分中增加iknet模块
            pred_local_pose, rotation_global_r6d, pred_joint_position = model(pred_local_pose, rotation_global_r6d,
                                                                              pred_joint_position, input_feat,
                                                                              pred_betas)

            pred_joint_position_head_centered = pred_joint_position - pred_joint_position[:, :, 15:16] + gt_positions[:,:, 15:16]
            gt_positions_head_centered = gt_positions

            loss_root, loss_ee, loss_ee_reg, loss_vel, total_loss = (
                loss_func(pred_local_pose, pred_joint_position_head_centered, pred_betas, gt_local_pose,
                          gt_positions_head_centered, gt_betas))

            total_loss.backward()
            optimizer.step()

            one_epoch_loss_root.append(loss_root.item())
            one_epoch_loss_ee.append(loss_ee.item())
            one_epoch_loss_ee_reg.append(loss_ee_reg.item())
            one_epoch_loss_vel.append(loss_vel.item())
            one_epoch_total_loss.append(total_loss.item())

        batch_time = time.time() - batch_start
        one_epoch_loss_root = torch.tensor(one_epoch_loss_root).mean().item()
        one_epoch_loss_ee = torch.tensor(one_epoch_loss_ee).mean().item()
        one_epoch_loss_ee_reg = torch.tensor(one_epoch_loss_ee_reg).mean().item()
        one_epoch_loss_vel = torch.tensor(one_epoch_loss_vel).mean().item()
        one_epoch_total_loss = torch.tensor(one_epoch_total_loss).mean().item()

        lr_scheduler.step(round(one_epoch_total_loss * 10000))

        train_summary_writer.add_scalar(
            'train_epoch_loss/loss_total', one_epoch_total_loss, epoch)
        train_summary_writer.add_scalar(
            'train_epoch_loss/loss_ee', one_epoch_loss_ee, epoch)
        train_summary_writer.add_scalar(
            'train_epoch_loss/loss_ee_reg', one_epoch_loss_ee_reg, epoch)
        train_summary_writer.add_scalar(
            'train_epoch_loss/loss_vel', one_epoch_loss_vel, epoch)

        epoch_info = {
            'type': 'train',
            'epoch': epoch + 1,
            'time': round(batch_time, 5),
            'loss_total': round(float(one_epoch_total_loss), 5),
            'loss_root': round(float(one_epoch_loss_root), 3),
            'loss_ee': round(float(one_epoch_loss_ee), 3),
            'loss_ee_reg': round(float(one_epoch_loss_ee_reg), 3),
            'loss_vel': round(float(one_epoch_loss_vel), 3),
        }
        LOG.info(epoch_info)

        if one_epoch_total_loss < best_train_loss:
            LOG.info("Saving model with best train loss in epoch {}".format(epoch + 1))
            filename = os.path.join(out_dir, "epoch_with_best_trainloss.pt")
            if os.path.exists(filename):
                os.remove(filename)
            model.save(epoch, filename)
            best_train_loss = one_epoch_total_loss

        if epoch % configs.train_config.val_interval == 0:
            model.eval()
            one_epoch_loss_root, one_epoch_loss_ee, one_epoch_loss_ee_reg, one_epoch_loss_vel, one_epoch_total_loss = [], [], [], [], []
            position_error_, local_pose_error_ = [], []
            position_error_upper, local_pose_error_upper = [], []
            with torch.no_grad():
                for batch_idx, (input_feat,pred_local_pose, pred_betas, rotation_global_r6d, pred_joint_position, gt_local_pose, gt_positions,gt_betas) in tqdm(enumerate(test_loader)):
                    # 在这个部分中增加iknet模块
                    pred_local_pose, rotation_global_r6d, pred_joint_position = model(pred_local_pose,rotation_global_r6d,pred_joint_position,input_feat, pred_betas)

                    pred_joint_position_head_centered = pred_joint_position - pred_joint_position[:, :,
                                                                              15:16] + gt_positions[:, :, 15:16]
                    gt_positions_head_centered = gt_positions

                    loss_root, loss_ee, loss_ee_reg, loss_vel, total_loss = (
                        loss_func(pred_local_pose, pred_joint_position_head_centered, pred_betas, gt_local_pose,
                                  gt_positions_head_centered, gt_betas))

                    pos_error_ = torch.mean(torch.sqrt(
                        torch.sum(
                            torch.square(gt_positions_head_centered - pred_joint_position_head_centered), axis=-1
                        )
                    ))
                    position_error_.append(pos_error_.item() * METERS_TO_CENTIMETERS)

                    pred_local_pose_aa = utils_transform.sixd2aa(pred_local_pose.reshape(-1, 6).detach()).reshape(-1,
                                                                                                                  22 * 3)
                    gt_local_pose_aa = utils_transform.sixd2aa(gt_local_pose.reshape(-1, 6).detach()).reshape(-1,
                                                                                                              22 * 3)
                    diff = gt_local_pose_aa - pred_local_pose_aa
                    diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
                    diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
                    rot_error = torch.mean(torch.absolute(diff))
                    local_pose_error_.append(rot_error.item() * RADIANS_TO_DEGREES)

                    # ---------------------------------------------------------------------------------------------
                    upper_indexes = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
                    pos_error_upper = torch.mean(torch.sqrt(
                        torch.sum(
                            torch.square(
                                gt_positions_head_centered[:, :, upper_indexes] - pred_joint_position_head_centered[:,
                                                                                  :, upper_indexes]), axis=-1
                        )
                    ))
                    position_error_upper.append(pos_error_upper.item() * METERS_TO_CENTIMETERS)

                    pred_local_pose_aa = utils_transform.sixd2aa(pred_local_pose.reshape(-1, 6).detach()).reshape(-1,
                                                                                                                  22,
                                                                                                                  3)[:,
                                         upper_indexes].reshape(-1, 14 * 3)
                    gt_local_pose_aa = utils_transform.sixd2aa(gt_local_pose.reshape(-1, 6).detach()).reshape(-1, 22,
                                                                                                              3)[:,
                                       upper_indexes].reshape(-1, 14 * 3)

                    diff = gt_local_pose_aa - pred_local_pose_aa
                    diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
                    diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
                    rot_error_upper = torch.mean(torch.absolute(diff))
                    local_pose_error_upper.append(rot_error_upper.item() * RADIANS_TO_DEGREES)


                    one_epoch_loss_root.append(loss_root.item())
                    one_epoch_loss_ee.append(loss_ee.item())
                    one_epoch_loss_ee_reg.append(loss_ee_reg.item())
                    one_epoch_loss_vel.append(loss_vel.item())
                    one_epoch_total_loss.append(total_loss.item())

            one_epoch_loss_root = torch.tensor(one_epoch_loss_root).mean().item()
            one_epoch_loss_ee = torch.tensor(one_epoch_loss_ee).mean().item()
            one_epoch_loss_ee_reg = torch.tensor(one_epoch_loss_ee_reg).mean().item()
            one_epoch_loss_vel = torch.tensor(one_epoch_loss_vel).mean().item()
            one_epoch_total_loss = torch.tensor(one_epoch_total_loss).mean().item()

            position_error_ = torch.tensor(position_error_).mean().item()
            local_pose_error_ = torch.tensor(local_pose_error_).mean().item()
            position_error_upper = torch.tensor(position_error_upper).mean().item()
            local_pose_error_upper = torch.tensor(local_pose_error_upper).mean().item()

            train_summary_writer.add_scalar(
                'train_epoch_loss/loss_total', one_epoch_total_loss, epoch)
            train_summary_writer.add_scalar(
                'train_epoch_loss/loss_ee', one_epoch_loss_ee, epoch)
            train_summary_writer.add_scalar(
                'train_epoch_loss/loss_ee_reg', one_epoch_loss_ee_reg, epoch)
            train_summary_writer.add_scalar(
                'train_epoch_loss/loss_vel', one_epoch_loss_vel, epoch)

            epoch_info = {
                'type': 'val',
                'epoch': epoch + 1,
                'loss_total': round(float(one_epoch_total_loss), 3),
                'loss_root': round(float(one_epoch_loss_root), 3),
                'loss_ee': round(float(one_epoch_loss_ee), 3),
                'loss_ee_reg': round(float(one_epoch_loss_ee_reg), 3),
                'loss_vel': round(float(one_epoch_loss_vel), 3),
            }
            LOG.info(epoch_info)

            model.train()

            if one_epoch_total_loss < best_test_loss:
                LOG.info("Saving model with lowest test loss in epoch {}".format(epoch + 1))
                filename = os.path.join(out_dir, "epoch_with_best_testloss.pt")
                if os.path.exists(filename):
                    os.remove(filename)
                model.save(epoch, filename)
                best_test_loss = one_epoch_total_loss

            if position_error_ < best_position:
                best_position = position_error_
                LOG.info("*************************************************************************")
                LOG.info("Lowest MPJPE {} in epoch {}".format(best_position, epoch + 1))
            else:
                LOG.info("MPJPE {} in epoch {}".format(position_error_, epoch + 1))

            if local_pose_error_ < best_local_pose:
                best_local_pose = local_pose_error_
                LOG.info("*************************************************************************")
                LOG.info("Lowest MPJRE_(including root) {} in epoch {}".format(best_local_pose, epoch + 1))
            else:
                LOG.info("MPJRE_(including root) {} in epoch {}".format(local_pose_error_, epoch + 1))

            LOG.info("Upper MPJPE {} in epoch {}".format(position_error_upper, epoch + 1))
            LOG.info("Upper MPJRE_(including root) {} in epoch {}".format(local_pose_error_upper, epoch + 1))
