import numpy as np
import torch
from monai.losses import DiceLoss
from x_unet import XUnet

from monai.transforms import (
    AddChanneld, Compose, LoadImaged, RandCropByPosNegLabeld,
    ToTensord, NormalizeIntensityd, RandFlipd,
    RandRotate90d, RandShiftIntensityd, RandAffined, RandSpatialCropd,
    RandScaleIntensityd)

PARAMS = dict(

    # trainining
    n_epochs=60,
    accumulated_batch_size=16,
    batch_size=2,
    
    optimizer=torch.optim.RAdam,
    optimizer_params=dict(lr=1e-3),
    scheduler=torch.optim.lr_scheduler.OneCycleLR,
    scheduler_params=dict(max_lr=1e-2, div_factor=10, final_div_factor=100, pct_start=0.1),
    monitor=None,

    # loss
    loss='weighted sum of dice and focal',
    gamma_focal=2.0,
    dice_weight=0.5,
    focal_weight=1.0,

    # validation
    sw_batch_size=2,
    roi_size=(48, 48, 48),
    thresh=0.4,
    iou_thresh=0.25,
    n_jobs=4,
    fracs_num_points=50,
    fracs_multiplier=4,
    
    # data
    path_train_data=["data/shifts_ms_pt1/msseg/train/flair", "data/shifts_ms_pt2/best/train/flair"],
    path_train_gts=["data/shifts_ms_pt1/msseg/train/gt", "data/shifts_ms_pt2/best/train/gt"],
    path_devin_data=["data/shifts_ms_pt1/msseg/dev_in/flair", "data/shifts_ms_pt2/best/dev_in/flair"],
    path_devin_gts=["data/shifts_ms_pt1/msseg/dev_in/gt", "data/shifts_ms_pt2/best/dev_in/gt"],
    path_devin_brain_masks=["data/shifts_ms_pt1/msseg/dev_in/fg_mask", "data/shifts_ms_pt2/best/dev_in/fg_mask"],
    path_evalin_data=["data/shifts_ms_pt1/msseg/eval_in/flair", "data/shifts_ms_pt2/best/eval_in/flair"],
    path_evalin_gts=["data/shifts_ms_pt1/msseg/eval_in/gt", "data/shifts_ms_pt2/best/eval_in/gt"],
    path_evalin_brain_masks=["data/shifts_ms_pt1/msseg/eval_in/fg_mask", "data/shifts_ms_pt2/best/eval_in/fg_mask"],

    num_workers=40,

    # logging
    val_interval=1,
    threshold=0.4,
    tb_logs='./runs',
    exp_name='xunet-radam-onecycle',
    ckpt_monitor='val-dev_in/dice_loss',

    # initialisation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    seed=42,

    num_gpus=2,

    # model
    model_name='XUNet',
    model_params=dict(dim = 8,
                      frame_kernel_size = 3,                 # set this to greater than 1;
                      channels = 1,
                      out_dim=2,
                      attn_dim_head = 4,
                      attn_heads = 2,
                      dim_mults = (1, 2),
                      num_blocks_per_stage = (1, 1),
                      num_self_attn_per_stage = (0, 1),
                      nested_unet_depths = (0,0),     # nested unet depths, from unet-squared paper
                      consolidate_upsample_fmaps = False,     # whether to consolidate outputs from all upsample blocks, used in unet-squared paper
                      weight_standardize = True
    )
)

model = XUnet(**PARAMS["model_params"])


def loss_function(outputs, labels):

    dice_fn = DiceLoss(to_onehot_y=True, 
                       softmax=True, sigmoid=False,
                       include_background=False)
    ce_fn = torch.nn.CrossEntropyLoss(reduction='none')

    # Dice loss
    dice_loss = dice_fn(outputs, labels)

    # Focal loss
    ce = ce_fn(outputs, torch.squeeze(labels, dim=1))
    pt = torch.exp(-ce)
    focal_loss = (1 - pt)**PARAMS["gamma_focal"] * ce 
    focal_loss = torch.mean(focal_loss)

    loss = PARAMS["dice_weight"] * dice_loss + PARAMS["focal_weight"] * focal_loss

    return {'loss': loss, 'dice_loss': dice_loss.detach().cpu(), 'focal_loss': focal_loss.detach().cpu()}


def get_train_transforms():
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"], nonzero=True),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandCropByPosNegLabeld(keys=["image", "label"],
                                   label_key="label", image_key="image",
                                   spatial_size=(128, 128, 128), num_samples=1,
                                   pos=4, neg=1),
            RandSpatialCropd(keys=["image", "label"],
                             roi_size=(96, 96, 96),
                             random_center=True, random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=(0, 1, 2)),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(1, 2)),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'),
                        prob=1.0, spatial_size=(48, 48, 48),
                        rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
                        scale_range=(0.1, 0.1, 0.1), padding_mode='border'),
            ToTensord(keys=["image", "label"]),
        ]
    )
