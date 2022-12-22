import argparse
import os
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import joblib

import torch
from torch import nn
from monai.inferers import sliding_window_inference
# from monai.losses import DiceLoss

from sklearn.metrics import auc

from metrics import *
from data_load import get_train_dataloader, get_val_dataloader, remove_connected_components
from uncertainty import ensemble_uncertainties_classification

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

import importlib


def compute_metrics(labels, outputs, threshold, iou_threshold, n_jobs=None):
    metrics = dict()

    gt = np.squeeze(labels.cpu().numpy())

    seg = nn.Softmax(dim=1)(outputs).cpu().numpy()
    seg = np.squeeze(seg[:,1])
    seg[seg >= threshold] = 1
    seg[seg < threshold] = 0

    metrics['dice'] = dice_metric(ground_truth=gt.flatten(), predictions=seg.flatten())
    metrics['nDSC'] = dice_norm_metric(ground_truth=gt.flatten(), predictions=seg.flatten())
    metrics['IoU'] = intersection_over_union(mask1=gt, mask2=seg)

    # parallel_backend = None
    # if n_jobs:
    #     parallel_backend = joblib.Parallel(n_jobs=n_jobs)
    # metrics['f1_lesion'] = lesion_f1_score(ground_truth=gt.flatten(), predictions=seg.flatten(), IoU_threshold=iou_threshold,
    #                                        parallel_backend=parallel_backend)

    return metrics

def get_fracs_retained(num_points=200):
    # Significant class imbalance means it is important to use logspacing between values
    # so that it is more granular for the higher retention fractions
    fracs_retained = np.log(np.arange(num_points + 1)[1:])
    fracs_retained /= np.amax(fracs_retained)

    return fracs_retained

def compute_ndsc_rc(labels, outputs, brain_mask, threshold):
    gt = np.squeeze(labels.cpu().numpy())

    seg = nn.Softmax(dim=1)(outputs).cpu().numpy()
    seg = np.squeeze(seg[0,1])
    all_outputs = np.asarray([seg])
    seg[seg >= threshold] = 1
    seg[seg < threshold] = 0
    seg = np.squeeze(seg)
    seg = remove_connected_components(seg)

    brain_mask = np.squeeze(brain_mask)

    # compute reverse mutual information uncertainty map
    uncs_map = ensemble_uncertainties_classification(np.concatenate((np.expand_dims(all_outputs, axis=-1),
                                                                     np.expand_dims(1. - all_outputs, axis=-1)),
                                                     axis=-1))['reverse_mutual_information']
    
    # compute metrics
    ndsc_rc = ndsc_retention_curve(ground_truth=gt[brain_mask == 1].flatten(),
                                   predictions=seg[brain_mask == 1].flatten(),
                                   uncertainties=uncs_map[brain_mask == 1].flatten(),
                                   fracs_retained=get_fracs_retained())

    return ndsc_rc



class LitModule(pl.LightningModule):
    def __init__(self, params, model, loss_function):
        super().__init__()

        self.save_hyperparameters(ignore=['model', 'loss_function'])

        self.params = params
        self.model = model
        self.loss_function = loss_function

        # self.dice_loss = DiceLoss(to_onehot_y=True, 
        #                           softmax=True, sigmoid=False,
        #                           include_background=False)
        # self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self):
        self.logger.log_hyperparams(self.hparams, metrics=dict.fromkeys(['hp/train_loss', 'hp/train_dice', 'hp/train_ndsc',
                                                                         'hp/val_dice', 'hp/val_ndsc'], 0))

    def configure_optimizers(self):
        optimizer = self.params["optimizer"](self.model.parameters(), **self.params["optimizer_params"])
            
        if self.params["scheduler"]:
            if type(self.params["scheduler"]).__name__ == "OneCycleLR":
                scheduler = self.params["scheduler"](optimizer, total_steps=self.trainer.estimated_stepping_batches,
                                                     **self.params["scheduler_params"])
                scheduler = {"scheduler": scheduler, "interval" : "step"}
            
            else:
                scheduler = self.params["scheduler"](optimizer, **self.params["scheduler_params"])
                
            optimizer_dict = {'optimizer': optimizer,
                              'lr_scheduler': scheduler,
                              'monitor': self.params["monitor"]}
                
            return optimizer_dict
        
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = (batch["image"], batch["label"].type(torch.long))

        outputs = self.forward(inputs)

        loss_dict = self.loss_function(outputs, labels)
        self.log('train/loss', loss_dict['loss'])

        metrics_dict = compute_metrics(labels, outputs.detach(), self.params["thresh"], self.params["iou_thresh"], self.params["n_jobs"])
        loss_dict.update(metrics_dict)

        return loss_dict

    def training_epoch_end(self, outputs):
        for k in outputs[0].keys():
            if not k == 'loss':
                total_metric = np.stack([x[k].detach().cpu().numpy() if isinstance(x[k], torch.Tensor) else x[k] for x in outputs]).mean()
                self.logger.experiment.add_scalar(f'train/{k}', total_metric, self.current_epoch)

        total_loss = np.stack([x['loss'].detach().cpu().numpy() for x in outputs]).mean()
        total_dice = np.stack([x['dice'] for x in outputs]).mean()
        total_ndsc = np.stack([x['nDSC'] for x in outputs]).mean()
        self.logger.log_metrics({'hp/train_loss': total_loss, 'hp/train_dice': total_dice, 'hp/train_ndsc': total_ndsc}, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        val_inputs, val_labels, val_brain_mask = (batch["image"], batch["label"], batch["brain_mask"])

        val_outputs = sliding_window_inference(val_inputs, self.params["roi_size"], 
                                               self.params["sw_batch_size"], 
                                               self.model, mode='gaussian')

        metrics_dict = compute_metrics(val_labels, val_outputs, self.params["thresh"], self.params["iou_thresh"], self.params["n_jobs"])
        self.log('val/dice_loss', 1-metrics_dict['dice'].sum().item())

        ndsc_rc = compute_ndsc_rc(val_labels, val_outputs, val_brain_mask.cpu(), self.params["thresh"])
        metrics_dict.update({'ndsc_rc': ndsc_rc})

        return metrics_dict

    def validation_epoch_end(self, outputs):
        for k in list(outputs[0])[:-1]:
            total_metric = np.stack([x[k] for x in outputs]).mean()
            self.logger.experiment.add_scalar(f'val/{k}', total_metric, self.current_epoch)

        total_dice = np.stack([x['dice'] for x in outputs]).mean()
        total_ndsc = np.stack([x['nDSC'] for x in outputs]).mean()
        self.logger.log_metrics({'hp/val_dice': total_dice, 'hp/val_ndsc': total_ndsc}, self.current_epoch)

        # Normalised Dice Coefficient (nDSC) retention curve
        ndsc_rc = [x['ndsc_rc'] for x in outputs]
        ndsc_rc = np.asarray(ndsc_rc)
        y = np.mean(ndsc_rc, axis=0)

        ndsc_auc = 1. - auc(get_fracs_retained(), y)
        self.logger.experiment.add_scalar('val/nDSC-AUC', ndsc_auc, self.current_epoch)

        fig = plt.figure(figsize=(10,7))
        plt.plot(get_fracs_retained(), y, label=f"nDSC R-AUC: {ndsc_auc:.4f}")
        plt.xlabel("Retention Fraction")
        plt.ylabel("nDSC")
        plt.xlim([0.0, 1.01])
        plt.legend()
        self.logger.experiment.add_figure('val/nDSC-rc', fig, self.current_epoch)
        plt.close()
        


def main(params_path, params_module):
    pl.seed_everything(params_module.PARAMS["seed"], workers=True)

    ''' Initialise dataloaders '''
    train_loader = get_train_dataloader(flair_path=params_module.PARAMS["path_train_data"], 
                                        gts_path=params_module.PARAMS["path_train_gts"], 
                                        num_workers=params_module.PARAMS["num_workers"],
                                        batch_size=params_module.PARAMS["batch_size"])
    val_loader = get_val_dataloader(flair_path=params_module.PARAMS["path_val_data"], 
                                    gts_path=params_module.PARAMS["path_val_gts"], 
                                    num_workers=params_module.PARAMS["num_workers"],
                                    bm_path=params_module.PARAMS["path_val_brain_masks"])

    ''' Initialise the model '''
    model = params_module.model
    loss_function = params_module.loss_function

    lit_module = LitModule(params_module.PARAMS, model, loss_function)

    tb_logger = TensorBoardLogger(save_dir=params_module.PARAMS["tb_logs"],
                                  name=params_module.PARAMS["exp_name"],
                                  default_hp_metric=False)
    
    model_checkpoint_dir = os.path.join(params_module.PARAMS["tb_logs"], params_module.PARAMS["exp_name"],
                                        'version_' + str(tb_logger.version), 'models')
    if not os.path.exists(model_checkpoint_dir):
        os.makedirs(model_checkpoint_dir)
    checkpoint_callback = ModelCheckpoint(verbose=True,
                                          save_top_k=2,
                                          monitor=params_module.PARAMS["ckpt_monitor"],
                                          dirpath=model_checkpoint_dir,
                                          filename='epoch={epoch:02d}-{step}-dice_loss={val/dice_loss:.5f}',
                                          auto_insert_metric_name=False)

    trainer = pl.Trainer(num_sanity_val_steps=0,
                         accelerator='gpu',
                         devices=params_module.PARAMS["num_gpus"],
                         strategy="DP",
                         max_epochs=params_module.PARAMS["n_epochs"],
                         precision=16,
                         logger=tb_logger,
                         callbacks=[LearningRateMonitor(logging_interval='step'),
                                    checkpoint_callback],
                        #  benchmark=True,
                         deterministic=True,
                         accumulate_grad_batches=max(1,params_module.PARAMS["accumulated_batch_size"]//params_module.PARAMS["batch_size"]),
                         log_every_n_steps=2,
                         check_val_every_n_epoch=params_module.PARAMS["val_interval"])

    with open(params_path) as f:
        write_params = f.read()
    trainer.logger.experiment.add_text(params_path, write_params)

    print('Saving logs to:', model_checkpoint_dir[:-6])

    trainer.fit(lit_module, train_loader, val_loader)            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get all command line arguments.')
    parser.add_argument('params', type=str, help='Path to parameters py file')
    args = parser.parse_args()

    # load params file as module
    spec = importlib.util.spec_from_file_location("params", args.params)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)

    main(args.params, params)
