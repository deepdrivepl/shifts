import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss

PARAMS = dict(

    # trainining
    n_epochs=300,
    accumulated_batch_size=16,
    batch_size=1,
    
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
    sw_batch_size=8,
    roi_size=(96, 96, 96),
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
    num_workers=20,

    # logging
    val_interval=1,
    threshold=0.4,
    tb_logs='./runs',
    exp_name='baseline-radam-onecycle',
    ckpt_monitor='val-dev_in/dice_loss',

    # initialisation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    seed=42,
    num_gpus=2,

    # model
    model_name='UNet',
    model_params=dict(spatial_dims=3,
                      in_channels=1,
                      out_channels=2,
                      channels=(32, 64, 128, 256, 512),
                      strides=(2, 2, 2, 2),
                      num_res_units=0)

)

model = UNet(**PARAMS["model_params"])


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

    return {'loss': loss, 'dice_loss': dice_loss, 'focal_loss': focal_loss}
