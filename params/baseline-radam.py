import torch
from monai.networks.nets import UNet

PARAMS = {

    # trainining
    'n_epochs': 300,
    'accumulate_grad_batches': None,
    'batch_size': 2,
    
    'optimizer': torch.optim.RAdam,
    'optimizer_params': {'lr': 1e-5},
    'scheduler': None,
    'scheduler_params': None,

    # loss
    'gamma_focal': 2.0,
    'dice_weight': 0.5,
    'focal_weight': 1.0,

    # validation
    'sw_batch_size': 8,
    'roi_size': (96, 96, 96),
    'thresh': 0.4,
    
    # data
    'path_train_data': "/mnt/12TB/projects/shifts2022/data/shifts_ms_pt1/msseg/train/flair",
    'path_train_gts': "/mnt/12TB/projects/shifts2022/data/shifts_ms_pt1/msseg/train/gt",
    'path_val_data': "/mnt/12TB/projects/shifts2022/data/shifts_ms_pt1/msseg/eval_in/flair",
    'path_val_gts': "/mnt/12TB/projects/shifts2022/data/shifts_ms_pt1/msseg/eval_in/gt",
    'num_workers': 2,

    # logging
    # 'val_interval': 5,
    'threshold': 0.4,
    'tb_logs': './runs',
    'exp_name': 'baseline-radam',

    # initialisation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    'seed': 1,
    'num_gpus': 1,

    # model
    'model_name': 'UNet',
    'model_params': {'spatial_dims': 3,
                     'in_channels': 1,
                     'out_channels': 2,
                     'channels': (32, 64, 128, 256, 512),
                     'strides': (2, 2, 2, 2),
                     'num_res_units': 0}

}

model = UNet(**PARAMS["model_params"])