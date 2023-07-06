import os
import sys
sys.path.append('../include/OpticalFlow/GM-Flow-master')

#activate python env

PYTHON_VE_PATH = "../gmflow-env"
# Optionally, activate virtual environment
if PYTHON_VE_PATH != "":
    ve_path = os.path.join(PYTHON_VE_PATH, 'bin', 'activate_this.py')
    exec(open(ve_path).read(), {'__file__': ve_path})
    print("activate python env!")
print("import torch!")
import torch
print("import torch end!")
print ("py version==",sys.version)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2
print("import cv2 end!")
import argparse
import numpy as np
import os
from data import build_train_dataset
from gmflow.gmflow import GMFlow
from loss import flow_loss_func
from evaluate import (validate_chairs, validate_things, validate_sintel, validate_kitti,
                      create_sintel_submission, create_kitti_submission, inference_on_dir)

from utils.logger import Logger
from utils import misc
from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed
from utils.utils import InputPadder, compute_out_of_boundary_mask

device= 'cuda'
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', default='tmp', type=str,
                    help='where to save the training log and models')
parser.add_argument('--stage', default='chairs', type=str,
                    help='training stage')
parser.add_argument('--image_size', default=[384, 512], type=int, nargs='+',
                    help='image size for training')
parser.add_argument('--padding_factor', default=16, type=int,
                    help='the input should be divisible by padding_factor, otherwise do padding')

parser.add_argument('--max_flow', default=400, type=int,
                    help='exclude very large motions during training')
parser.add_argument('--val_dataset', default=['chairs'], type=str, nargs='+',
                    help='validation dataset')
parser.add_argument('--with_speed_metric', action='store_true',
                    help='with speed metric when evaluation')

# resume pretrained model or resume training
parser.add_argument('--resume', default="../include/OpticalFlow/GM-Flow-master/gmflow_sintel-0c07dcb3.pth", type=str,
                    help='resume from pretrain model for finetuing or resume from terminated training')
parser.add_argument('--strict_resume', action='store_true')
parser.add_argument('--no_resume_optimizer', action='store_true')

# GMFlow model
parser.add_argument('--num_scales', default=1, type=int,
                    help='basic gmflow model uses a single 1/8 feature, the refinement uses 1/4 feature')
parser.add_argument('--feature_channels', default=128, type=int)
parser.add_argument('--upsample_factor', default=8, type=int)
parser.add_argument('--num_transformer_layers', default=6, type=int)
parser.add_argument('--num_head', default=1, type=int)
parser.add_argument('--attention_type', default='swin', type=str)
parser.add_argument('--ffn_dim_expansion', default=4, type=int)

parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                    help='number of splits in attention')
parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                    help='correlation radius for matching, -1 indicates global matching')
parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                    help='self-attention radius for flow propagation, -1 indicates global attention')
# loss
parser.add_argument('--gamma', default=0.9, type=float,
                    help='loss weight')
# evaluation
parser.add_argument('--eval', action='store_true')
parser.add_argument('--save_eval_to_file', action='store_true')
parser.add_argument('--evaluate_matched_unmatched', action='store_true')

# inference on a directory
parser.add_argument('--inference_dir', default="../include/OpticalFlow/GM-Flow-master/demo/sintel_market_1", type=str)
parser.add_argument('--inference_size', default=None, type=int, nargs='+',
                    help='can specify the inference size')
parser.add_argument('--dir_paired_data', action='store_true',
                    help='Paired data in a dir instead of a sequence')
parser.add_argument('--save_flo_flow', action='store_true')
parser.add_argument('--pred_bidir_flow', action='store_true',
                    help='predict bidirectional flow')
parser.add_argument('--fwd_bwd_consistency_check', action='store_true',
                    help='forward backward consistency check with bidirection flow')

# predict on sintel and kitti test set for submission
parser.add_argument('--submission', action='store_true',
                    help='submission to sintel or kitti test sets')
parser.add_argument('--output_path', default='../include/OpticalFlow/GM-Flow-master/output', type=str,
                    help='where to save the prediction results')
parser.add_argument('--save_vis_flow', action='store_true',
                    help='visualize flow prediction as .png image')
parser.add_argument('--no_save_flo', action='store_true',
                    help='not save flow as .flo')

# distributed training
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

parser.add_argument('--count_time', action='store_true',
                    help='measure the inference time on sintel')
args = parser.parse_args()
model = GMFlow(feature_channels=args.feature_channels,
               num_scales=args.num_scales,
               upsample_factor=args.upsample_factor,
               num_head=args.num_head,
               attention_type=args.attention_type,
               ffn_dim_expansion=args.ffn_dim_expansion,
               num_transformer_layers=args.num_transformer_layers,
               ).to(device)
model_without_ddp = model
start_epoch = 0
start_step = 0
if args.resume:
    print('Load checkpoint: %s' % args.resume)
    loc = 'cuda:{}'.format(args.local_rank)
    checkpoint = torch.load(args.resume, map_location=loc)

    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

    model_without_ddp.load_state_dict(weights, strict=args.strict_resume)
    model.eval()
print("loaded model!")
def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    colorwheel = make_color_wheel()
    print("make_color_wheel")
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img

def display(flow):
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8
    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    print("display")
    u[idxUnknow] = 0
    v[idxUnknow] = 0
    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)
    print("compute_color")
    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0
    from PIL import Image
    img = Image.fromarray(np.uint8(img))
    img.show()

@torch.no_grad()   #very important!!!
def execute(img1, img2):   # img1  curr  img2 last one
    #print("111")
    image1 = np.array(img1).astype(np.uint8)
    image2 = np.array(img2).astype(np.uint8)
    image1 = image1[..., :3]
    image2 = image2[..., :3]
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
    padder = InputPadder(image1.shape, padding_factor=16)
    image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
    #print("222")
    results_dict = model(image1, image2,
                         attn_splits_list=[2],
                         corr_radius_list=[-1],
                         prop_radius_list=[-1],
                         pred_bidir_flow=False,
                         )
    global flow_up_u
    global flow_up_v
    flow_pr = results_dict['flow_preds'][-1]
    flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
    flow_up_u = flow[:, :, 0]
    #print('flow_up_u:')
    #print(flow_up_u)
    flow_up_v = flow[:, :, 1]
    # print(flow_up_u)
    # print(flow_up_v)
    #display(flow)

