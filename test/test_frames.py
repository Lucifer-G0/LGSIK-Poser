import math

import torch
import os
import argparse
import random
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.utils_config import load_config, add_log, LearningRateLambda
from dataset.dataloader_shape import load_data, TrainDataset, TestDataset
from torch.utils.data import DataLoader
from model.hmd_imu_model import HMDIMUModel
from model.loss import PoseJointLoss
from runner.training_loop import train_loop


from tqdm import tqdm
from utils import utils_transform

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
METERS_TO_CENTIMETERS = 100.0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        help="Path, where config file is stored", default="./options/test_config.yaml")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    configs = load_config(args.config)
    out_dir, LOG, train_summary_writer = add_log(configs)
    dst_file = os.path.join(out_dir, os.path.basename(args.config))
    os.system('cp {} {}'.format(args.config, dst_file))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    '''
    # ----------------------------------------
    # seed
    # ----------------------------------------
    '''
    seed = configs.manual_seed
    if seed is None:
        seed = random.randint(1, 10000)
    LOG.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    LOG.info("creating test dataloader...")
    test_datas = load_data(configs.dataset_path, "test",
                           input_motion_length=configs.input_motion_length)
    test_dataset = TestDataset(test_datas, configs.compatible_inputs,
                               configs.input_motion_length, 1)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=configs.train_config.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    LOG.info("There are {} video sequence, {} item and {} batch in the test set.".format(
        len(test_datas),
        len(test_dataset),
        len(test_dataloader)
    )
    )

    LOG.info("creating model...")
    model = HMDIMUModel(configs, device)
    if configs.resume_model is not None:
        model.load(configs.resume_model)
        LOG.info(f"successfully resume checkpoint in {configs.resume_model}")

    position_error_ = torch.zeros(configs.input_motion_length,device=device)
    model.eval()
    with torch.no_grad():
            for _, (input_feat, gt_local_pose, gt_global_pose, gt_positions, gt_betas, gt_sparse_offsets) in tqdm(
                    enumerate(test_dataloader)):
                    input_feat, gt_local_pose, gt_global_pose, gt_positions, gt_betas, gt_sparse_offsets = input_feat.to(device).float(), \
                            gt_local_pose.to(device).float(), gt_global_pose.to(device).float(), \
                            gt_positions.to(device).float(), gt_betas.to(device).float(), gt_sparse_offsets.to(device).float()
                    if len(input_feat.shape) == 4:
                            input_feat = torch.flatten(input_feat, start_dim=0, end_dim=1)
                            gt_positions = torch.flatten(gt_positions, start_dim=0, end_dim=1)
                            gt_sparse_offsets = torch.flatten(gt_sparse_offsets, start_dim=0, end_dim=1)


                    pred_local_pose, pred_betas, rotation_global_r6d, pred_joint_position = model(input_feat,gt_sparse_offsets)

                    pred_joint_position_head_centered = pred_joint_position - pred_joint_position[:, :,15:16] + gt_positions[:, :, 15:16]
                    gt_positions_head_centered = gt_positions

                    pos_error_ = torch.mean(torch.sqrt(
                            torch.sum(
                                    torch.square(gt_positions_head_centered - pred_joint_position_head_centered),
                                    axis=-1
                            )
                    ),dim=(0,2))
                    position_error_ += pos_error_

            print("MPJPE: ",torch.mean(position_error_)/len(test_dataloader) * METERS_TO_CENTIMETERS)
            print(position_error_/len(test_dataloader) * METERS_TO_CENTIMETERS)


if __name__ == "__main__":
    main()
