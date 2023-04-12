"""
Computation of performance metrics (nDSC, lesion F1 score, nDSC R-AUC) 
for an ensemble of models.
Metrics are displayed in console.
Additionaly save predicted and true lesion volume and lesion count.
"""

import os
import argparse
from tqdm import tqdm
import torch
from joblib import Parallel
from monai.inferers import sliding_window_inference
from monai.transforms import CropForeground

import numpy as np
from data_load import remove_connected_components, get_val_dataloader, get_val_transforms
from metrics import dice_norm_metric, lesion_f1_score, ndsc_aac_metric
from uncertainty import ensemble_uncertainties_classification
from scipy import ndimage

from collections import OrderedDict
import importlib


parser = argparse.ArgumentParser(description='Get all command line arguments.')
# model
parser.add_argument('--path_model', type=str, default='',
                    help='Specify the path to the trained model')
parser.add_argument('--path_params', type=str, default='',
                    help='Specify the path to the model params')
# data
parser.add_argument('--path_data', nargs='+', required=True,
                    help='Specify the path to the directory with FLAIR images')
parser.add_argument('--path_gts', nargs='+', required=True,
                    help='Specify the path to the directory with ground truth binary masks')
parser.add_argument('--path_bm', nargs='+', required=True,
                    help='Specify the path to the directory with brain masks')
# parallel computation
parser.add_argument('--num_workers', type=int, default=1,
                    help='Number of workers to preprocess images')
parser.add_argument('--n_jobs', type=int, default=1,
                    help='Number of parallel workers for F1 score computation')
# hyperparameters
parser.add_argument('--threshold', type=float, default=0.35,
                    help='Probability threshold')
parser.add_argument('--uncertainty', type=str, default='entropy_of_expected',
                    help='Uncertainty method for nDSC retention curve calculation')
# save
parser.add_argument('--path_save', type=str, required=True,
                    help='Specify the path to the directory where the metrics will be saved')
parser.add_argument('--dont_save', action='store_true', default=False,
                    help='Don\'t save any arrays, only display metrics')


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def main(args):
    device = get_default_device()
    torch.multiprocessing.set_sharing_strategy('file_system')

    os.makedirs(args.path_save, exist_ok=True)

    spec = importlib.util.spec_from_file_location("params", args.path_params)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)

    '''' Initialise dataloaders '''
    val_loader = get_val_dataloader(flair_paths=args.path_data,
                                    gts_paths=args.path_gts,
                                    num_workers=args.num_workers,
                                    bm_paths=args.path_bm,
                                    transforms=params.get_val_transforms)

    ''' Load trained model '''
    model = params.model

    roi_size = params.PARAMS['roi_size']
    sw_batch_size = 4
    
    checkpoint = torch.load(args.path_model, map_location=device)
    checkpoint = OrderedDict({k.replace('model.', '', 1):v for k,v in checkpoint['state_dict'].items()})
    model.load_state_dict(checkpoint)

    if device == torch.device('cuda'):
        model.half()

    model.to(device)
    model.eval()

    def predictor(inputs):
        if inputs.max() > 0.2:
            return model(inputs)
        return torch.zeros((sw_batch_size, 2, 64, 64, 64), dtype=inputs.dtype, layout=inputs.layout, device=inputs.device)

    act = torch.nn.Softmax(dim=1)

    crop = CropForeground()

    ndsc, f1, ndsc_aac = [], [], []
    gt_vol, pred_vol, gt_count, pred_count = [], [], [], []

    ''' Evaluatioin loop '''
    with Parallel(n_jobs=args.n_jobs) as parallel_backend:
        with torch.no_grad():
            for count, batch_data in tqdm(enumerate(val_loader)):
                inputs, gt, brain_mask = (
                    batch_data["image"],
                    batch_data["label"].cpu().numpy(),
                    batch_data["brain_mask"].cpu().numpy()
                )
                # gt lesion count and volume
                labeled_gt, _ = ndimage.label(gt)
                label_list = np.unique(labeled_gt)
                gt_count.append(len(label_list))
                gt_vol.append(np.sum(gt))

                inputs = crop(inputs[0][0])
                inputs = inputs.unsqueeze(0).unsqueeze(0)

                if device == torch.device('cuda'):
                    inputs = inputs.half()

                # get predictions
                outputs = sliding_window_inference(inputs.to(device), roi_size,
                                                   sw_batch_size, predictor,
                                                   mode='gaussian')

                outputs = act(outputs)
                outputs.applied_operations = inputs.applied_operations
                outputs = crop.inverse(outputs[0,1])
                outputs = outputs.cpu().numpy().astype(np.float32)

                # obtain binary segmentation mask
                seg = outputs.copy()
                seg[seg >= args.threshold] = 1
                seg[seg < args.threshold] = 0
                seg = np.squeeze(seg)
                seg = remove_connected_components(seg)

                # lesion count and volume
                labeled_seg, _ = ndimage.label(seg)
                label_list = np.unique(labeled_seg)
                pred_count.append(len(label_list))
                pred_vol.append(np.sum(seg))
                
                gt = np.squeeze(gt)
                brain_mask = np.squeeze(brain_mask)

                # compute reverse mutual information uncertainty map
                uncs_map = ensemble_uncertainties_classification(np.concatenate(
                    (np.expand_dims(outputs, axis=(0,-1)),
                     np.expand_dims(1. - outputs, axis=(0,-1))),
                    axis=-1))[args.uncertainty]

                # compute metrics
                ndsc += [dice_norm_metric(ground_truth=gt, predictions=seg)]
                f1 += [lesion_f1_score(ground_truth=gt,
                                       predictions=seg,
                                       IoU_threshold=0.5,
                                       parallel_backend=parallel_backend)]
                ndsc_aac += [ndsc_aac_metric(ground_truth=gt[brain_mask == 1].flatten(),
                                             predictions=seg[brain_mask == 1].flatten(),
                                             uncertainties=uncs_map[brain_mask == 1].flatten(),
                                             parallel_backend=parallel_backend)]

    for arr, name in zip([ndsc, f1, ndsc_aac, gt_vol, pred_vol, gt_count, pred_count],
                         ['nDSC', 'f1', 'nDSC_R-AUC', 'gt_volume', 'pred_volume', 'gt_count', 'pred_count']):
        arr = np.asarray(arr)
        if not args.dont_save:
            np.save(os.path.join(args.path_save, name + '.npy'), arr)

    print(args.path_data[0].split('/')[-2])
    print(f"nDSC:\t{np.mean(ndsc):.4f} +- {np.std(ndsc):.4f}")
    print(f"Lesion F1 score:\t{np.mean(f1):.4f} +- {np.std(f1):.4f}")
    print(f"nDSC R-AUC:\t{np.mean(ndsc_aac):.4f} +- {np.std(ndsc_aac):.4f}")


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
