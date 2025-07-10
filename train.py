import torch
import os
import argparse
import random
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.utils_config import load_config, add_log
from dataset.dataloader_shape import load_data, TrainDataset, TestDataset
from torch.utils.data import DataLoader
from model.hmd_imu_model import HMDIMUModel
from model.loss import PoseJointLoss
from runner.training_loop import train_loop


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        help="Path, where config file is stored", default="./options/train_config.yaml")
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

    '''
    # ----------------------------------------
    # creat dataloader
    # ----------------------------------------
    '''
    LOG.info("creating train dataloader...")
    train_datas = load_data(configs.dataset_path, "train",
                            input_motion_length=configs.input_motion_length)
    train_dataset = TrainDataset(train_datas, configs.compatible_inputs,
                                 configs.input_motion_length, configs.train_dataset_repeat_times)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=configs.train_config.batch_size,	# configs.train_config.batch_size
        shuffle=True,
        num_workers=configs.train_config.num_workers,
        drop_last=True,
        persistent_workers=False,
    )
    LOG.info("There are {} video sequence, {} item and {} batch in the training set.".format(
        len(train_datas),
        len(train_dataset),
        len(train_dataloader)
    )
    )

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

    loss_func = PoseJointLoss(configs.loss.loss_type,
                              configs.loss.root_orientation_weight,
                              configs.loss.local_pose_weight,
                              configs.loss.global_pose_weight,
                              configs.loss.joint_position_weight,
                              configs.loss.smooth_loss_weight,
                              configs.loss.shape_loss_weight
                              ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), 0.001)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2,eps=1e-7)

    train_loop(configs, device, model, loss_func, optimizer, lr_scheduler, train_dataloader, test_dataloader, LOG,
               train_summary_writer, out_dir)


if __name__ == "__main__":
    main()
