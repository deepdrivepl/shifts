"""
Metrics used for validation during training and evaluation: 
Dice Score, Normalised Dice score, Lesion F1 score and nDSC R-AAC.
"""
import numpy as np
from functools import partial
from scipy import ndimage
from collections import Counter
from joblib import Parallel, delayed
from sklearn import metrics
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F



def dice_metric(ground_truth, predictions):
    """
    Compute Dice coefficient for a single example.
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [W, H, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [W, H, D].
    Returns:
      Dice coefficient overlap (`float` in [0.0, 1.0])
      between `ground_truth` and `predictions`.
    """
    # Calculate intersection and union of y_true and y_predict
    intersection = np.sum(predictions * ground_truth)
    union = np.sum(predictions) + np.sum(ground_truth)

    # Calcualte dice metric
    if intersection == 0.0 and union == 0.0:
        dice = 1.0
    else:
        dice = (2. * intersection) / union

    return dice


def dice_metric_pytorch(ground_truth, predictions):
    # Calculate intersection and union of y_true and y_predict
    intersection = torch.sum(predictions * ground_truth)
    union = torch.sum(predictions) + torch.sum(ground_truth)

    # Calcualte dice metric
    if intersection == 0.0 and union == 0.0:
        dice = 1.0
    else:
        dice = ((2. * intersection) / union).cpu()

    return dice


def dice_norm_metric(ground_truth, predictions):
    """
    Compute Normalised Dice Coefficient (nDSC), 
    False positive rate (FPR),
    False negative rate (FNR) for a single example.
    
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      Normalised dice coefficient (`float` in [0.0, 1.0]),
      False positive rate (`float` in [0.0, 1.0]),
      False negative rate (`float` in [0.0, 1.0]),
      between `ground_truth` and `predictions`.
    """

    # Reference for normalized DSC
    r = 0.001
    # Cast to float32 type
    gt = ground_truth.astype("float32")
    seg = predictions.astype("float32")
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 1.0
    else:
        if np.sum(gt) == 0:
            k = 1.0
        else:
            k = (1 - r) * np.sum(gt) / (r * (len(gt.flatten()) - np.sum(gt)))
        tp = np.sum(seg[gt == 1])
        fp = np.sum(seg[gt == 0])
        fn = np.sum(gt[seg == 0])
        fp_scaled = k * fp
        dsc_norm = 2. * tp / (fp_scaled + 2. * tp + fn)
        return dsc_norm


def get_tp_fn_fp(ground_truth, predictions):
    tp = ((ground_truth == predictions) & (ground_truth == 1))
    fn = ((ground_truth != predictions) & (ground_truth == 1))
    fp = ((ground_truth != predictions) & (ground_truth == 0))

    return tp, fn, fp


def dice_norm_metric_pytorch(ground_truth, predictions):
    # Reference for normalized DSC
    r = 0.001

    im_sum = torch.sum(predictions) + torch.sum(ground_truth)
    if im_sum == 0:
        return 1.0
    else:
        if torch.sum(ground_truth) == 0:
            k = 1.0
            tp = 0
            fn = 0
            fp = ((ground_truth != predictions) & (ground_truth == 0)).sum()

        else:
            k = (1 - r) * torch.sum(ground_truth) / (r * (torch.numel(ground_truth) - torch.sum(ground_truth)))
            tp, fn, fp = get_tp_fn_fp(ground_truth, predictions)
            tp, fn, fp = tp.sum(), fn.sum(), fp.sum()

        fp_scaled = k * fp
        dsc_norm = 2. * tp / (fp_scaled + 2. * tp + fn)
        return dsc_norm.cpu()


def ndsc_aac_metric(ground_truth, predictions, uncertainties, parallel_backend=None):
    """
    Compute area above Normalised Dice Coefficient (nDSC) retention curve for 
    one subject. `ground_truth`, `predictions`, `uncertainties` - are flattened 
    arrays of correponding 3D maps within the foreground mask only.
    
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H * W * D]. 
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H * W * D].
      uncertainties:  `numpy.ndarray`, voxel-wise uncertainties,
                     with shape [H * W * D].
      parallel_backend: `joblib.Parallel`, for parallel computation
                     for different retention fractions.
    Returns:
      nDSC R-AAC (`float` in [0.0, 1.0]).
    """

    def compute_dice_norm(frac_, preds_, gts_, N_):
        pos = int(N_ * frac_)
        curr_preds = preds if pos == N_ else np.concatenate(
            (preds_[:pos], gts_[pos:]))
        return dice_norm_metric(gts_, curr_preds)

    if parallel_backend is None:
        parallel_backend = Parallel(n_jobs=1)

    ordering = uncertainties.argsort()
    gts = ground_truth[ordering].copy()
    preds = predictions[ordering].copy()
    N = len(gts)

    # # Significant class imbalance means it is important to use logspacing between values
    # # so that it is more granular for the higher retention fractions
    fracs_retained = np.log(np.arange(200 + 1)[1:])
    fracs_retained /= np.amax(fracs_retained)

    process = partial(compute_dice_norm, preds_=preds, gts_=gts, N_=N)
    dsc_norm_scores = np.asarray(
        parallel_backend(delayed(process)(frac)
                         for frac in fracs_retained)
    )

    return 1. - metrics.auc(fracs_retained, dsc_norm_scores)


def ndsc_retention_curve(ground_truth, predictions, uncertainties, fracs_retained, parallel_backend=None):
    """
    Compute Normalised Dice Coefficient (nDSC) retention curve.
    
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H * W * D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H * W * D].
      uncertainties:  `numpy.ndarray`, voxel-wise uncertainties,
                     with shape [H * W * D].
      fracs_retained:  `numpy.ndarray`, array of increasing valies of retained 
                       fractions of most certain voxels, with shape [N].
      parallel_backend: `joblib.Parallel`, for parallel computation
                     for different retention fractions.
    Returns:
      (y-axis) nDSC at each point of the retention curve (`numpy.ndarray` with shape [N]).
    """

    def compute_dice_norm(frac_, preds_, gts_, N_):
        pos = int(N_ * frac_)
        curr_preds = preds if pos == N_ else np.concatenate(
            (preds_[:pos], gts_[pos:]))
        return dice_norm_metric(gts_, curr_preds)

    # if parallel_backend is None:
    #     parallel_backend = Parallel(n_jobs=1)

    ordering = uncertainties.argsort()
    gts = ground_truth[ordering].copy()
    preds = predictions[ordering].copy()
    N = len(gts)

    # process = partial(compute_dice_norm, preds_=preds, gts_=gts, N_=N)
    # dsc_norm_scores = np.asarray(
    #     parallel_backend(delayed(process)(frac)
    #                      for frac in fracs_retained)
    # )
    dsc_norm_scores = np.asarray([compute_dice_norm(preds_=preds, gts_=gts, N_=N, frac_=frac) for frac in fracs_retained])

    return dsc_norm_scores


def ndsc_retention_curve_pytorch(ground_truth, predictions, uncertainties, fracs_retained, device):
    def compute_dice_norm(frac_, preds_, gts_, N_):
        pos = int(N_ * frac_)
        curr_preds = preds if pos == N_ else torch.concatenate(
            (preds_[:pos], gts_[pos:]))
        return dice_norm_metric_pytorch(gts_, curr_preds)

    ordering = uncertainties.argsort()
    gts = ground_truth[ordering].clone()
    preds = predictions[ordering].clone()
    N = len(gts)

    dsc_norm_scores = torch.tensor([compute_dice_norm(preds_=preds, gts_=gts, N_=N, frac_=frac) for frac in fracs_retained],
                                   device=device)

    return dsc_norm_scores.cpu()


def intersection_over_union(mask1, mask2):
    """
    Compute IoU for 2 binary masks.
    
    Args:
      mask1: `numpy.ndarray`, binary mask.
      mask2:  `numpy.ndarray`, binary mask of the same shape as `mask1`.
    Returns:
      Intersection over union between `mask1` and `mask2` (`float` in [0.0, 1.0]).
    """
    return np.sum(mask1 * mask2) / np.sum(mask1 + mask2 - mask1 * mask2)


def intersection_over_union_pytorch(mask1, mask2):
    if torch.sum(mask1) + torch.sum(mask2) == 0:
        return 1.0
        
    return (torch.sum(mask1 * mask2) / torch.sum(mask1 + mask2 - mask1 * mask2)).cpu()


def lesion_f1_score(ground_truth, predictions, IoU_threshold=0.25, parallel_backend=None):
    """
    Compute lesion-scale F1 score.
    
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
      IoU_threshold: `float` in [0.0, 1.0], IoU threshold for max IoU between 
                     predicted and ground truth lesions to classify them as
                     TP, FP or FN.
      parallel_backend: `joblib.Parallel`, for parallel computation
                     for different retention fractions.
    Returns:
      Intersection over union between `mask1` and `mask2` (`float` in [0.0, 1.0]).
    """

    def get_tp_fp(label_pred, mask_multi_pred, mask_multi_gt):
        mask_label_pred = (mask_multi_pred == label_pred).astype(int)
        all_iou = [0.0]
        # iterate only intersections
        for int_label_gt in np.unique(mask_multi_gt * mask_label_pred):
            if int_label_gt != 0.0:
                mask_label_gt = (mask_multi_gt == int_label_gt).astype(int)
                all_iou.append(intersection_over_union(
                    mask_label_pred, mask_label_gt))
        max_iou = max(all_iou)
        if max_iou >= IoU_threshold:
            return 'tp'
        else:
            return 'fp'

    def get_fn(label_gt, mask_multi_pred, mask_multi_gt):
        mask_label_gt = (mask_multi_gt == label_gt).astype(int)
        all_iou = [0]
        for int_label_pred in np.unique(mask_multi_pred * mask_label_gt):
            if int_label_pred != 0.0:
                mask_label_pred = (mask_multi_pred ==
                                   int_label_pred).astype(int)
                all_iou.append(intersection_over_union(
                    mask_label_pred, mask_label_gt))
        max_iou = max(all_iou)
        if max_iou < IoU_threshold:
            return 1
        else:
            return 0

    mask_multi_pred_, n_les_pred = ndimage.label(predictions)
    mask_multi_gt_, n_les_gt = ndimage.label(ground_truth)

    if parallel_backend is None:
        parallel_backend = Parallel(n_jobs=1)

    process_fp_tp = partial(get_tp_fp, mask_multi_pred=mask_multi_pred_,
                            mask_multi_gt=mask_multi_gt_)

    tp_fp = parallel_backend(delayed(process_fp_tp)(label_pred)
                             for label_pred in np.unique(mask_multi_pred_) if label_pred != 0)
    counter = Counter(tp_fp)
    tp = float(counter['tp'])
    fp = float(counter['fp'])

    process_fn = partial(get_fn, mask_multi_pred=mask_multi_pred_,
                         mask_multi_gt=mask_multi_gt_)

    fn = parallel_backend(delayed(process_fn)(label_gt)
                          for label_gt in np.unique(mask_multi_gt_) if label_gt != 0)
    fn = float(np.sum(fn))

    f1 = 1.0 if tp + 0.5 * (fp + fn) == 0.0 else tp / (tp + 0.5 * (fp + fn))

    return f1


class nDSC_Loss(_Loss):
    # Ref.: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/35899
    
    def __init__(self, r=0.001):
        super().__init__(reduction='mean')
        self.r = r

    def forward(self, predictions, ground_truth):
        
        ground_truth = torch.squeeze(ground_truth).to(predictions.dtype)

        predictions = torch.squeeze(predictions[:,1])

        im_sum = torch.sum(predictions) + torch.sum(ground_truth)

        if im_sum == 0:
            return 1.0 - 1.0
    
        else:
            if torch.sum(ground_truth) == 0:
                k = 1.0
                tp_loss = 0
                fn_loss = 0

            else:
                k = (1 - self.r) * torch.sum(ground_truth) / (self.r * (torch.numel(ground_truth) - torch.sum(ground_truth)))
                tp_loss = (ground_truth * (1 - F.binary_cross_entropy_with_logits(predictions, ground_truth))).sum()
                fn_loss = (ground_truth * F.binary_cross_entropy_with_logits(predictions, ground_truth)).sum()
                
            fp_loss = ((1 - ground_truth) * F.binary_cross_entropy_with_logits(predictions, ground_truth)).sum()

            fp_scaled = k * fp_loss
            dsc_norm = 2. * tp_loss / (fp_scaled + 2. * tp_loss + fn_loss)
            return 1 - dsc_norm
