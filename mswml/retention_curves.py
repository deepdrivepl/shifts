"""
Build nDSC retention curve plot.
"""

import argparse
import os
import torch
from joblib import Parallel
from monai.inferers import sliding_window_inference

import numpy as np
from data_load import remove_connected_components, get_val_dataloader, get_val_transforms
from metrics import ndsc_retention_curve
from uncertainty import ensemble_uncertainties_classification
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', font_scale=1.5)
from sklearn import metrics
from tqdm import tqdm

from collections import OrderedDict, defaultdict

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
# other
parser.add_argument('--path_save', type=str, required=True,
                    help='Specify the path to the directory where retention curves will be saved')
parser.add_argument('--plot_title', type=str, default='')


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

    act = torch.nn.Softmax(dim=1)

    # Significant class imbalance means it is important to use logspacing between values
    # so that it is more granular for the higher retention fractions
    fracs_retained = np.log(np.arange(200 + 1)[1:])
    fracs_retained /= np.amax(fracs_retained)

    # uncertainties = ['confidence', 'entropy_of_expected', 'expected_entropy', 'mutual_information', 'epkl', 'reverse_mutual_information']
    uncs_dict = defaultdict(list)

    ''' Evaluatioin loop '''
    with Parallel(n_jobs=args.n_jobs) as parallel_backend:
        with torch.no_grad():
            for count, batch_data in tqdm(enumerate(val_loader)):
                inputs, gt, brain_mask = (
                    batch_data["image"],
                    batch_data["label"].cpu().numpy(),
                    batch_data["brain_mask"].cpu().numpy()
                )

                if device == torch.device('cuda'):
                    inputs = inputs.half()

                # get predictions
                outputs = sliding_window_inference(inputs.to(device), roi_size,
                                                   sw_batch_size, model,
                                                   mode='gaussian')
                outputs = act(outputs).cpu().numpy().astype(np.float32)
                outputs = np.squeeze(outputs[0, 1])

                # obtain binary segmentation mask
                seg = outputs.copy()
                seg[seg >= args.threshold] = 1
                seg[seg < args.threshold] = 0
                seg = np.squeeze(seg)
                seg = remove_connected_components(seg)

                gt = np.squeeze(gt)
                brain_mask = np.squeeze(brain_mask)

                # compute reverse mutual information uncertainty map
                uncs = ensemble_uncertainties_classification(np.concatenate(
                    (np.expand_dims(outputs, axis=(0,-1)),
                     np.expand_dims(1. - outputs, axis=(0,-1))),
                    axis=-1))

                # compute metrics
                for k in uncs.keys():
                    ndsc_rc = ndsc_retention_curve(ground_truth=gt[brain_mask == 1].flatten(),
                                                   predictions=seg[brain_mask == 1].flatten(),
                                                   uncertainties=uncs[k][brain_mask == 1].flatten(),
                                                   fracs_retained=fracs_retained,
                                                   parallel_backend=parallel_backend)
                    uncs_dict[k].append(ndsc_rc)

    plt.figure(figsize=(20,12))

    for k in uncs_dict.keys():
        ndsc_rc = np.asarray(uncs_dict[k])
        y = np.mean(ndsc_rc, axis=0)
        np.save(os.path.join(args.path_save, 'nDSC_rc_' + k + '.npy'), y)

        plt.plot(fracs_retained, y, label=f"{k}: {1. - metrics.auc(fracs_retained, y):.4f}")
        
    plt.xlabel("Retention Fraction")
    plt.ylabel("nDSC")
    plt.xlim([0.0, 1.0])
    plt.title(args.plot_title)
    plt.legend(title='nDSC R-AUC')
    plt.savefig(os.path.join(args.path_save, 'nDSC_rc.jpg'))
    plt.clf()


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
