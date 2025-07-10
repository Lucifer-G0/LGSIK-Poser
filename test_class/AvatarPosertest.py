'''
# --------------------------------------------
# main testing code
# --------------------------------------------
# AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing (ECCV 2022)
# https://github.com/eth-siplab/AvatarPoser
# Jiaxi Jiang (jiaxi.jiang@inf.ethz.ch)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
'''

import os.path
import argparse
import time

import torch
from thop import profile
from utils import utils_option as option
from human_body_prior.body_model.body_model import BodyModel

save_animation = False
resolution = (800,800)

def main(json_path='options/test_avatarposer.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)

    paths = (path for key, path in opt['path'].items() if 'pretrained' not in key)
    if isinstance(paths, str):
        if not os.path.exists(paths):
            os.makedirs(paths)
    else:
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_netG'] = init_path_G
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


    from models.network import AvatarPoser as net
    model = net(input_dim=opt_net['input_dim'],
               output_dim=opt_net['output_dim'],
               num_layer=opt_net['num_layer'],
               embed_dim=opt_net['embed_dim'],
               nhead=opt_net['nhead'],
               body_model=body_model,
               device=device).to(device)

    # 构造 dummy 输入
    input_feat = torch.randn(1, 40, opt_net['input_dim']).to(device)  # 你模型的主输入

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
    main()
