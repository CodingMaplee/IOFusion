import os
import sys
sys.path.append('../include/InstanceSeg/SOLO')
PYTHON_VE_PATH = "../solo-env"
if PYTHON_VE_PATH != "":
    ve_path = os.path.join(PYTHON_VE_PATH, 'bin', 'activate_this.py')
    exec(open(ve_path).read(), {'__file__': ve_path})
    print("activate python env!")

import numpy as np
from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv

config_file = '../include/InstanceSeg/SOLO/configs/solov2/solov2_r50_fpn_8gpu_1x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../include/InstanceSeg/SOLO/checkpoints/epoch_805.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
def execute(image):
    h, w, _ = image.shape
    #output: width*height, with label in pixel
    global output
    output = np.zeros((h, w), dtype=np.int32, order='C')
    result = inference_detector(model, image)
    cur_result = result[0]
    if cur_result is None:
        return

    seg_label = cur_result[0]
    seg_label = seg_label.cpu().numpy().astype(np.uint8)
    cate_label = cur_result[1]
    cate_label = cate_label.cpu().numpy()
    score = cur_result[2].cpu().numpy()
    vis_inds = score > 0.25
    seg_label = seg_label[vis_inds]
    num_mask = seg_label.shape[0]
    cate_label = cate_label[vis_inds]
    cate_score = score[vis_inds]

    for idx in range(num_mask):
        idx = -(idx+1)
        cur_mask = seg_label[idx, :, :]
        cur_mask = mmcv.imresize(cur_mask, (w, h))
        cur_mask = (cur_mask > 0.5).astype(np.uint8)
        if cur_mask.sum() == 0:
            continue
        cur_cate = cate_label[idx]
        positive_mask = cur_mask > 0
        output[positive_mask] = cur_cate + 1




    #print(result)
    #show_result_ins(image, result, model.CLASSES, score_thr=0.25, out_file="demo_out.jpg")


