"""
Perform inference and save 3D Nifti images of
- predicted probability maps (saved to "*pred_prob.nii.gz" files),
- binary segmentation maps predicted obtained by thresholding of predictions and 
removing all connected components smaller than 9 voxels (saved to "pred_seg.nii.gz"),
- uncertainty maps for reversed mutual information measure (saved to "uncs_rmi.nii.gz").
"""

import argparse
import os
import re
import torch
from monai.inferers import sliding_window_inference
# from monai.networks.nets import UNet
from monai.data import write_nifti
import numpy as np
from data_load import remove_connected_components, get_flair_dataloader, get_val_transforms
from uncertainty import ensemble_uncertainties_classification

from x_unet import XUnet
from monai.networks.nets import UNet
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Get all command line arguments.')
# save options
parser.add_argument('--path_pred', type=str, required=True,
                    help='Specify the path to the directory to store predictions')
# model
parser.add_argument('--path_model', type=str, default='',
                    help='Specify the dir to all the trained models')
# data
parser.add_argument('--path_data', nargs='+', required=True,
                    help='Specify the path to the directory with FLAIR images')
parser.add_argument('--path_bm', nargs='+', required=True,
                    help='Specify the path to the directory with brain masks')
# parallel computation
parser.add_argument('--num_workers', type=int, default=10,
                    help='Number of workers to preprocess images')
# hyperparameters
parser.add_argument('--threshold', type=float, default=0.35,
                    help='Probability threshold')
parser.add_argument('--uncertainty', type=str, default='entropy_of_expected',
                    help='Uncertainty method for nDSC retention curve calculation')


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def main(args):
    os.makedirs(args.path_pred, exist_ok=True)
    device = get_default_device()
    torch.multiprocessing.set_sharing_strategy('file_system')

    '''' Initialise dataloaders '''
    val_loader = get_flair_dataloader(flair_paths=args.path_data,
                                      num_workers=args.num_workers,
                                      bm_paths=args.path_bm,
                                      transforms=get_val_transforms)

    ''' Load trained models  '''
    model = XUnet(dim = 64,
                 frame_kernel_size = 3,
                 channels = 1,
                 out_dim = 2,
                 attn_dim_head = 32,
                 attn_heads = 8,
                 dim_mults = (1, 2, 4, 8),
                 num_blocks_per_stage = (2, 2, 2, 2),
                 num_self_attn_per_stage = (0, 0, 0, 1),
                 nested_unet_depths = (5, 4, 2, 1),
                 consolidate_upsample_fmaps = True,
                 weight_standardize = False)
    roi_size = (64, 64, 64)
    sw_batch_size = 4

    # model = UNet(spatial_dims = 3,
    #              in_channels = 1,
    #              out_channels =2,
    #              channels = (32, 64, 128, 256, 512),
    #              strides = (2, 2, 2, 2),
    #              num_res_units = 0)
    # roi_size = (96, 96, 96)
    # sw_batch_size = 8

    checkpoint = torch.load(args.path_model, map_location=device)
    checkpoint = OrderedDict({k.replace('model.', '', 1):v for k,v in checkpoint['state_dict'].items()})
    model.load_state_dict(checkpoint)

    if device == torch.device('cuda'):
        model.half()

    model.to(device)
    model.eval()

    act = torch.nn.Softmax(dim=1)

    ''' Predictions loop '''
    with torch.no_grad():
        for count, batch_data in enumerate(val_loader):
            inputs = batch_data["image"].to(device)
            foreground_mask = batch_data["brain_mask"].numpy()[0, 0]

            if device == torch.device('cuda'):
                inputs = inputs.half()

            # get predictions
            outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model, mode='gaussian')

            outputs = act(outputs).cpu().numpy().astype(np.float32)
            outputs = np.squeeze(outputs[0, 1])

            # get image metadata
            original_affine = batch_data['image_meta_dict']['original_affine'][0]
            affine = batch_data['image_meta_dict']['affine'][0]
            spatial_shape = batch_data['image_meta_dict']['spatial_shape'][0]
            filename_or_obj = batch_data['image_meta_dict']['filename_or_obj'][0]
            filename_or_obj = os.path.basename(filename_or_obj)

            # save probability maps
            filename = re.sub("FLAIR_isovox.nii.gz", 'pred_prob.nii.gz',
                                filename_or_obj)
            filepath = os.path.join(args.path_pred, filename)
            write_nifti(outputs, filepath,
                        affine=original_affine,
                        target_affine=affine,
                        output_spatial_shape=spatial_shape)

            # obtain and save binary segmentation masks
            seg = outputs.copy()
            seg[seg >= args.threshold] = 1
            seg[seg < args.threshold] = 0
            seg = np.squeeze(seg)
            seg = remove_connected_components(seg)

            filename = re.sub("FLAIR_isovox.nii.gz", 'pred_seg.nii.gz',
                            filename_or_obj)
            filepath = os.path.join(args.path_pred, filename)
            write_nifti(seg, filepath,
                        affine=original_affine,
                        target_affine=affine,
                        mode='nearest',
                        output_spatial_shape=spatial_shape)

            # obtain and save uncertainty map
            uncs_map = ensemble_uncertainties_classification(np.concatenate(
                    (np.expand_dims(outputs, axis=(0,-1)),
                    np.expand_dims(1. - outputs, axis=(0,-1))),
                    axis=-1))[args.uncertainty]

            filename = re.sub("FLAIR_isovox.nii.gz", 'uncs.nii.gz',
                            filename_or_obj)
            filepath = os.path.join(args.path_pred, filename)
            write_nifti(uncs_map * foreground_mask, filepath,
                        affine=original_affine,
                        target_affine=affine,
                        output_spatial_shape=spatial_shape)



# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
