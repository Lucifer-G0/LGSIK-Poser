import os
import os.path
import argparse
import logging
import time

import torch
from thop import profile
from torch.utils.data import DataLoader

from human_body_prior.body_model.body_model import BodyModel
from utils import utils_logger
from utils import utils_option as option
from data.select_dataset import define_Dataset
from models.select_model import define_Model
from models.network import AvatarJLM


def main(opt, save_animation=False):
    paths = (path for key, path in opt['path'].items() if 'pretrained' not in key)
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)

    option.save(opt)
    opt = option.dict_to_nonedict(opt)

    opt_net = opt['netG']
    net_type = opt_net['net_type']
    device = torch.device('cuda' if opt['gpu_ids'] else 'cpu')
    support_dir = opt['support_dir']
    subject_gender = "male"
    bm_fname = os.path.join(support_dir, 'body_models/smplh/{}/model.npz'.format(subject_gender))
    dmpl_fname = os.path.join(support_dir, 'body_models/dmpls/{}/model.npz'.format(subject_gender))
    num_betas = 16  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters
    body_model = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(
        device)

    model = AvatarJLM(body_model=body_model,
                      input_dim=opt_net['input_dim'],
                      nhead=opt_net['nhead'],
                      embed_dim=opt_net['embed_dim'],
                      single_frame_feat_dim=opt_net['single_frame_feat_dim'],
                      joint_regressor_dim=opt_net['joint_regressor_dim'],
                      joint_embed_dim=opt_net['joint_embed_dim'],
                      mask_training=opt_net['mask_training'],
                      replace=opt_net['replace'],
                      position_token=opt_net['position_token'],
                      rotation_token=opt_net['rotation_token'],
                      input_token=opt_net['input_token']
                      ).to(device)

    # 构造 dummy 输入
    input_feat = torch.randn(1, 40, 22,18).to(device) # 你模型的主输入

    # 计算 FLOPs 和 参数
    flops, params = profile(model, inputs=(input_feat,))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Params: {params / 1e6:.2f} M")

    # 预热 GPU（重要！避免首次推理偏高）
    for _ in range(10):
        _ = model(input_feat)

    # 正式计时
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10000):  # 多次运行平均更稳定
        _ = model(input_feat)
    torch.cuda.synchronize()
    end_time = time.time()

    # 计算平均延迟
    avg_latency = (end_time - start_time) / 10000 * 1000  # 毫秒
    print(f"Latency: {avg_latency:.2f} ms per inference")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./options/opt_ajlm.json', help='Path to option JSON file.')
    parser.add_argument('--task', type=str, default='AvatarJLM', help='Experiment name.')
    parser.add_argument('--protocol', type=str, choices=['1', '2', '3', 'real'], default='1', help='Protocol.')
    parser.add_argument('--checkpoint', type=str, default="", help='Trained model weights.')
    parser.add_argument('--vis', action="store_true", help='Save animation.')
    args = parser.parse_args()
    opt = option.parse(args.opt, args, is_train=False)
    main(opt, args.vis)
