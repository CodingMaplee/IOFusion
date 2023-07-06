import os
import sys
sys.path.append('../include/OpticalFlow/RAFT-master/core')

#activate python env

PYTHON_VE_PATH = "../python-environment"
# Optionally, activate virtual environment
if PYTHON_VE_PATH != "":
    ve_path = os.path.join(PYTHON_VE_PATH, 'bin', 'activate_this.py')
    exec(open(ve_path).read(), {'__file__': ve_path})


import argparse
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
DEVICE = 'cuda'
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="../include/OpticalFlow/RAFT-master/models/raft-sintel.pth")
parser.add_argument('--path', type=str, default="../include/OpticalFlow/RAFT-master/demo-frames")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
args = parser.parse_args()

model = torch.nn.DataParallel(RAFT(args))

model.load_state_dict(torch.load(args.model))
model = model.module
model.to(DEVICE)
model.eval()

def viz(img, flo):
    #img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().detach().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    # img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    cv2.imshow('image', flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey(1)

@torch.no_grad()
def execute(img1, img2):   # img1  curr  img2 last one

    img1 = img1.astype(np.uint8)
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()

    img1 = img1[None].to(DEVICE)
    img2 = img2.astype(np.uint8)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
    img2 = img2[None].to(DEVICE)

    global flow_up_u
    global flow_up_v
    padder = InputPadder(img1.shape)

    image1, image2 = padder.pad(img1, img2)
    flow_low, flow_up = model(image1, image2, iters=30, test_mode=True)

    #print(flow_up)
    #print("222222222222222222")
    #print(flow_low)
    flow_up_u = flow_up[0][0].cpu().detach().numpy()

    flow_up_v = flow_up[0][1].cpu().detach().numpy()

    viz(img2, flow_up)

