"""
Adjust threshold based on nDSC/lesion F1 score.
"""

import argparse
import os
import torch
from joblib import Parallel
from monai.inferers import sliding_window_inference
from monai.transforms import CropForeground
import numpy as np
from data_load import remove_connected_components, get_val_dataloader, get_val_transforms
from metrics import dice_norm_metric, lesion_f1_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', font_scale=1.5)
from tqdm import tqdm

from collections import OrderedDict
import importlib

parser = argparse.ArgumentParser(description='Get all command line arguments.')
# model
parser.add_argument('--path_model', type=str, default='',
                    help='Specify the path to the checkpoint')
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
# other
parser.add_argument('--path_save', type=str, required=True,
                    help='Specify the path to the directory where plots and metrics will be saved')
parser.add_argument('--plot_title', type=str, default='',
                    help='Title of the generated plots')
parser.add_argument('--th_step', type=int, default=10,
                    help='Calculate metrics every th_step/100 threshold between 0 and 1')


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def main(args):
    os.makedirs(args.path_save, exist_ok=True)
    device = get_default_device()
    torch.multiprocessing.set_sharing_strategy('file_system')

    spec = importlib.util.spec_from_file_location("params", args.path_params)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)

    '''' Initialise dataloaders '''
    val_loader = get_val_dataloader(flair_paths=args.path_data,
                                    gts_paths=args.path_gts,
                                    num_workers=args.num_workers,
                                    bm_paths=args.path_bm,
                                    transforms=params.get_val_transforms)

    ''' Load trained model  '''
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

    thresholds = np.asarray(range(0, 100 + args.th_step, args.th_step)) / 100

    ndsc, f1 = [], []

    ''' Evaluatioin loop '''
    with Parallel(n_jobs=args.n_jobs) as parallel_backend:
        with torch.no_grad():
            for th in tqdm(thresholds):
                ndsc_th, f1_th = [], []
                for count, batch_data in enumerate(val_loader):
                    inputs, gt, brain_mask = (
                        batch_data["image"],
                        batch_data["label"].cpu().numpy(),
                        batch_data["brain_mask"].cpu().numpy()
                    )

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
                    seg = outputs.cpu().numpy().astype(np.float32)

                    # obtain binary segmentation mask
                    seg[seg >= th] = 1
                    seg[seg < th] = 0
                    seg = np.squeeze(seg)
                    seg = remove_connected_components(seg)

                    gt = np.squeeze(gt)
                    brain_mask = np.squeeze(brain_mask)

                    ndsc_th += [dice_norm_metric(ground_truth=gt, predictions=seg)]
                    f1_th += [lesion_f1_score(ground_truth=gt,
                                              predictions=seg,
                                              IoU_threshold=0.5,
                                              parallel_backend=parallel_backend)]
            
                ndsc.append(np.mean(ndsc_th))
                f1.append(np.mean(f1_th))

    ''' Save plots and metrics '''
    for metric, metric_name in zip([ndsc, f1], ['nDSC', 'f1']):
        metric = np.asarray(metric)
        np.save(os.path.join(args.path_save, metric_name + '_threshold.npy'), metric)

        plt.figure(figsize=(20,12))
        plt.plot(thresholds, metric)
        plt.xlabel("Threshold")
        plt.ylabel(metric_name)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])

        max_metric = np.max(metric)
        max_metric_th = thresholds[np.argmax(metric)]
        plt.text(0.005, max_metric + 0.01, f"Max {metric_name} = {max_metric:.4f}")
        plt.text(max_metric_th+0.005, 0.01, f"Threshold = {max_metric_th:.2f}")
        plt.plot([0,max_metric_th],[max_metric, max_metric], linestyle='dashed', c='black')
        plt.plot([max_metric_th,max_metric_th],[0, max_metric], linestyle='dashed', c='black')

        plt.title(args.plot_title)

        plt.savefig(os.path.join(args.path_save, metric_name + '_threshold.jpg'))
        plt.clf()


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
