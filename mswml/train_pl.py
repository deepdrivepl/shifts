import argparse
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import joblib

import torch
from torch import nn
from monai.inferers import sliding_window_inference
# from monai.losses import DiceLoss
from monai.visualize.utils import blend_images, matshow3d
from monai.visualize.img2tensorboard import plot_2d_or_3d_image

from sklearn.metrics import auc

from metrics import *
from data_load import get_train_dataloader, get_val_dataloader#, remove_connected_components
from uncertainty import ensemble_uncertainties_classification_pytorch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

import importlib


def compute_metrics(labels, outputs, threshold, iou_threshold, n_jobs=None):
    metrics = dict()
    with torch.no_grad():
        gt = torch.squeeze(labels)

        seg = postprocess_output(outputs, threshold)
        if seg.shape[1]>1:
            seg = torch.squeeze(seg[:,1])
        else:
            seg = torch.squeeze(seg)

        metrics['dice'] = dice_metric_pytorch(gt, seg)
        metrics['nDSC'] = dice_norm_metric_pytorch(gt, seg)
        metrics['IoU'] = intersection_over_union_pytorch(gt, seg)

    # parallel_backend = None
    # if n_jobs:
    #     parallel_backend = joblib.Parallel(n_jobs=n_jobs)
    # metrics['f1_lesion'] = lesion_f1_score(ground_truth=gt.flatten(), predictions=seg.flatten(), IoU_threshold=iou_threshold,
    #                                        parallel_backend=parallel_backend)

    return metrics

def get_fracs_retained(num_points=200, multiplier=1):
    # Significant class imbalance means it is important to use logspacing between values
    # so that it is more granular for the higher retention fractions
    fracs_retained = np.log((np.arange(num_points)*multiplier + 1))
    fracs_retained /= np.amax(fracs_retained)

    return fracs_retained

def get_fracs_retained_pytorch(num_points=200, multiplier=1):
    # Significant class imbalance means it is important to use logspacing between values
    # so that it is more granular for the higher retention fractions
    fracs_retained = torch.log((torch.arange(num_points)*multiplier + 1))
    
    fracs_retained /= torch.amax(fracs_retained)

    return fracs_retained

def compute_ndsc_rc(labels, outputs, brain_mask, threshold, device, fracs_num_points, fracs_multiplier):
    gt = torch.squeeze(labels)

    if outputs.shape[1] > 1:
        seg = nn.Softmax(dim=1)(outputs)
        seg = torch.squeeze(seg[0,1])
    else:
        seg = nn.Sigmoid()(outputs)
        seg = torch.squeeze(seg[0,0])
    
    all_outputs = seg.unsqueeze(0)
    seg[seg >= threshold] = 1
    seg[seg < threshold] = 0
    seg = torch.squeeze(seg)
    # seg = remove_connected_components(seg.cpu().numpy())
    # seg = torch.tensor(seg, device=device)

    brain_mask = torch.squeeze(brain_mask)

    # compute reverse mutual information uncertainty map
    uncs_map = ensemble_uncertainties_classification_pytorch(torch.concatenate((all_outputs.unsqueeze(-1),
                                                                               (1. - all_outputs).unsqueeze(-1)),
                                                             dim=-1))['reverse_mutual_information']

    brain_mask_cpu = brain_mask.cpu().numpy()
    ground_truth = torch.tensor(gt.cpu().numpy()[brain_mask_cpu == 1], device=device)
    predictions = torch.tensor(seg.cpu().numpy()[brain_mask_cpu == 1], device=device)
    uncertainties = torch.tensor(uncs_map.cpu().numpy()[brain_mask_cpu == 1], device=device)
    
    # def generate_masked_tensor(input, mask, fill=0):
    #     masked_tensor = torch.zeros(input.size()) + fill
    #     masked_tensor[mask] = input[mask]
    #     return masked_tensor

    # mask = torch.nonzero((brain_mask == 1).bool())
    # ground_truth=generate_masked_tensor(gt, mask)
    # predictions=generate_masked_tensor(seg, mask)
    # uncertainties=generate_masked_tensor(uncs_map, mask)

    # compute metrics
    ndsc_rc = ndsc_retention_curve_pytorch(ground_truth=ground_truth.flatten(),
                                           predictions=predictions.flatten(),
                                           uncertainties=uncertainties.flatten(),
                                           fracs_retained=get_fracs_retained_pytorch(fracs_num_points, fracs_multiplier),
                                           device=device)

    return ndsc_rc

def blend_imgs(img, label, pred):
    tp, fn, fp = get_tp_fn_fp(label, pred)

    out = blend_images(img, tp, alpha=0.5, cmap='Greens')
    out = blend_images(out, fn, alpha=0.5, cmap='summer')
    out = blend_images(out, fp, alpha=0.5, cmap='Reds')

    return out

def postprocess_output(outputs, threshold):
    if outputs.shape[1] > 1:
        seg = nn.Softmax(dim=1)(outputs)
    else:
        seg = nn.Sigmoid()(outputs)

    seg[seg >= threshold] = 1
    seg[seg < threshold] = 0

    return seg


class LitModule(pl.LightningModule):
    def __init__(self, params, model, loss_function):
        super().__init__()

        self.save_hyperparameters(ignore=['model', 'loss_function'])

        self.params = params
        self.model = model
        self.loss_function = loss_function

        self.dataloader_suffix = ['-dev_in', '-eval_in']

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self):
        self.logger.log_hyperparams(self.hparams, metrics=dict.fromkeys(['hp/train_loss', 'hp/train_dice', 'hp/train_ndsc',
                                                                         'hp/val-dev_in_dice', 'hp/val-dev_in_ndsc',
                                                                         'hp/val-eval_in_dice', 'hp/val-eval_in_ndsc'], 0))
        if self.global_rank == 0:
            print('Saving logs to:', self.trainer.log_dir)

    def configure_optimizers(self):
        optimizer = self.params["optimizer"](self.model.parameters(), **self.params["optimizer_params"])
            
        if self.params["scheduler"]:
            if self.params["scheduler"].__name__ == "OneCycleLR":
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

    # def on_before_optimizer_step(self, optimizer, optimizer_idx):
    #     if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
    #         for k, v in self.named_parameters():
    #             if len(v.grad):
    #                 self.logger.experiment.add_histogram(
    #                 tag=k, values=v.grad, global_step=self.trainer.global_step
    #             )

    def training_step(self, batch, batch_idx):
        inputs, labels = (batch["image"], batch["label"].type(torch.long))

        outputs = self.forward(inputs)

        loss_dict = self.loss_function(outputs, labels)
        self.log('train/loss', loss_dict['loss'], batch_size=self.params["batch_size"])
        
        metrics_dict = compute_metrics(labels.detach(), outputs.detach(), self.params["thresh"], self.params["iou_thresh"], self.params["n_jobs"])
        loss_dict.update(metrics_dict)

        # log images
        if batch_idx == 0:
            blended = blend_imgs(inputs[0], labels[0], 
                                 postprocess_output(outputs.detach(), self.params["thresh"])[0,outputs.shape[1]-1].unsqueeze(0))
            self.log_mosaic(blended, 'train', figsize=(14,14))

        return loss_dict

    def training_epoch_end(self, outputs):
        for k in outputs[0].keys():
            if not k == 'loss':
                #total_metric = torch.stack([x[k].detach() for x in outputs]).mean()
                # print(k,[x[k] for x in outputs])
                total_metric = np.mean([x[k] for x in outputs])
                self.logger.experiment.add_scalar(f'train/{k}', total_metric, self.current_epoch)

        total_loss = torch.stack([x['loss'].detach() for x in outputs]).mean()
        total_dice = np.mean([float(x['dice']) for x in outputs])
        total_ndsc = np.mean([float(x['nDSC']) for x in outputs])
        self.logger.log_metrics({'hp/train_loss': total_loss.item(), 'hp/train_dice': total_dice.item(),
                                 'hp/train_ndsc': total_ndsc.item()}, self.current_epoch)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        val_inputs, val_labels, val_brain_mask = (batch["image"], batch["label"], batch["brain_mask"])

        val_outputs = sliding_window_inference(val_inputs, self.params["roi_size"], 
                                               self.params["sw_batch_size"], 
                                               self.model, mode='gaussian')

        metrics_dict = compute_metrics(val_labels, val_outputs, self.params["thresh"], self.params["iou_thresh"], self.params["n_jobs"])
        self.log(f'val{self.dataloader_suffix[dataloader_idx]}/dice_loss', 1-metrics_dict['dice'].sum().item(),
                 batch_size=1, add_dataloader_idx=False, sync_dist=True)

        ndsc_rc = compute_ndsc_rc(val_labels, val_outputs, val_brain_mask, self.params["thresh"], device=self.device,
                                  fracs_num_points=self.params["fracs_num_points"], fracs_multiplier=self.params["fracs_multiplier"])
        metrics_dict.update({'ndsc_rc': ndsc_rc})

        # log images
        if batch_idx < self.params["num_images_val"]:
            blended = blend_imgs(val_inputs[0], val_labels[0], 
                                 postprocess_output(val_outputs.detach(), self.params["thresh"])[0,val_outputs.shape[1]-1].unsqueeze(0))
            study_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].rstrip('.nii.gz')
            try:
                self.log_mosaic(blended, f'val{self.dataloader_suffix[dataloader_idx]}-images/{study_name}', frames_per_row=6, crop=True)
            except Exception as e:
                print(e)

            if self.current_epoch % self.params["log_gif_interval"] == 0 and self.local_rank == 0:
                plot_2d_or_3d_image([blended], self.current_epoch, self.logger.experiment, max_channels=3,
                                     tag=f'val{self.dataloader_suffix[dataloader_idx]}-gif/{study_name}',
                                     max_frames=5)

        return metrics_dict

    def validation_epoch_end(self, outputs):
        for i, dataloader_outputs in enumerate(outputs):
            for k in list(dataloader_outputs[0])[:-1]:
                total_metric = torch.stack([x[k] for x in dataloader_outputs]).mean()
                self.logger.experiment.add_scalar(f'val{self.dataloader_suffix[i]}/{k}', total_metric.item(), self.current_epoch)

            total_dice = torch.stack([x['dice'] for x in dataloader_outputs]).mean()
            total_ndsc = torch.stack([x['nDSC'] for x in dataloader_outputs]).mean()
            self.logger.log_metrics({f'hp/val{self.dataloader_suffix[i]}_dice': total_dice.item(),
                                     f'hp/val{self.dataloader_suffix[i]}_ndsc': total_ndsc.item()}, self.current_epoch)

            # Normalised Dice Coefficient (nDSC) retention curve
            ndsc_rc = [x['ndsc_rc'].cpu().numpy() for x in dataloader_outputs]
            ndsc_rc = np.asarray(ndsc_rc)
            y = np.mean(ndsc_rc, axis=0)

            ndsc_auc = 1. - auc(get_fracs_retained(self.params["fracs_num_points"], self.params["fracs_multiplier"]), y)
            self.logger.experiment.add_scalar(f'val{self.dataloader_suffix[i]}/nDSC-AUC', ndsc_auc, self.current_epoch)

            fig = plt.figure(figsize=(10,7))
            plt.plot(get_fracs_retained(self.params["fracs_num_points"], self.params["fracs_multiplier"]), y,
                     label=f"nDSC R-AUC: {ndsc_auc:.4f}")
            plt.xlabel("Retention Fraction")
            plt.ylabel("nDSC")
            plt.xlim([0.0, 1.01])
            plt.ylim([0.0, 1.01])
            plt.legend()
            self.logger.experiment.add_figure(f'val{self.dataloader_suffix[i]}/nDSC-rc', fig, self.current_epoch)
            plt.close()

    def log_mosaic(self, blended, tag, frames_per_row=None, figsize=None, crop=False):
        if frames_per_row:
            num_rows = blended.shape[1] // frames_per_row + ((blended.shape[1] % frames_per_row) > 0)
            figsize = (30,num_rows*5)
            blended = blended[:,:-(blended.shape[1] % frames_per_row)]

        fig, _ = matshow3d(blended, vmin=0, vmax=1, figsize=figsize,
                           channel_dim=0, fill_value=1, frames_per_row=frames_per_row)
        self.logger.experiment.add_figure(tag, fig, self.current_epoch)
        plt.close()



def main(params_path, params_module):
    pl.seed_everything(params_module.PARAMS["seed"], workers=True)

    ''' Initialise dataloaders '''
    train_loader = get_train_dataloader(flair_paths=params_module.PARAMS["path_train_data"], 
                                        gts_paths=params_module.PARAMS["path_train_gts"], 
                                        num_workers=params_module.PARAMS["num_workers"],
                                        batch_size=params_module.PARAMS["batch_size"],
                                        transforms=params_module.get_train_transforms,
                                        cache_rate=params_module.PARAMS["cache_rate"],
                                        multiply=params_module.PARAMS["multiply_train"])
    val_loader_devin = get_val_dataloader(flair_paths=params_module.PARAMS["path_devin_data"], 
                                          gts_paths=params_module.PARAMS["path_devin_gts"], 
                                          num_workers=params_module.PARAMS["num_workers"],
                                          bm_paths=params_module.PARAMS["path_devin_brain_masks"],
                                          transforms=params_module.get_val_transforms,
                                          cache_rate=params_module.PARAMS["cache_rate"])
    val_loader_evalin = get_val_dataloader(flair_paths=params_module.PARAMS["path_evalin_data"], 
                                           gts_paths=params_module.PARAMS["path_evalin_gts"], 
                                           num_workers=params_module.PARAMS["num_workers"],
                                           bm_paths=params_module.PARAMS["path_evalin_brain_masks"],
                                           transforms=params_module.get_val_transforms,
                                           cache_rate=params_module.PARAMS["cache_rate"])

    ''' Initialise the model '''
    model = params_module.model
    loss_function = params_module.loss_function

    lit_module = LitModule(params_module.PARAMS, model, loss_function)

    tb_logger = TensorBoardLogger(save_dir=params_module.PARAMS["tb_logs"],
                                  name=params_module.PARAMS["exp_name"],
                                  default_hp_metric=False)
    
    model_checkpoint_dir = os.path.join(params_module.PARAMS["tb_logs"], params_module.PARAMS["exp_name"],
                                        'version_' + str(tb_logger.version), 'models')
    checkpoint_callback = ModelCheckpoint(verbose=True,
                                          save_top_k=2,
                                          monitor=params_module.PARAMS["ckpt_monitor"],
                                          dirpath=model_checkpoint_dir,
                                          filename='epoch={epoch:02d}-{step}-dice_loss={val-eval_in/dice_loss:.5f}',
                                          auto_insert_metric_name=False)

    profiler = None
    if 'profiler_fname' in params_module.PARAMS.keys():
        from pytorch_lightning.profilers import AdvancedProfiler
        profiler_logs_dir = os.path.join(params_module.PARAMS["tb_logs"], params_module.PARAMS["exp_name"],
                                        'version_' + str(tb_logger.version))
        profiler = AdvancedProfiler(dirpath=profiler_logs_dir, filename=params_module.PARAMS['profiler_fname'])

    trainer = pl.Trainer(num_sanity_val_steps=0,
                         accelerator='gpu',
                         devices=params_module.PARAMS["num_gpus"],
                         strategy=params_module.PARAMS["strategy"],
                         max_epochs=params_module.PARAMS["n_epochs"],
                         precision=params_module.PARAMS.get("precision", 16),
                         gradient_clip_val=params_module.PARAMS.get("grad_clip_value", 0.1),
                         track_grad_norm=2,
                         logger=tb_logger,
                         callbacks=[LearningRateMonitor(logging_interval='step'),
                                    checkpoint_callback],
                        #  benchmark=True,
                         # deterministic=True,  # CE Loss reduction error
                         accumulate_grad_batches=max(1,params_module.PARAMS["accumulated_batch_size"]//params_module.PARAMS["batch_size"]),
                         log_every_n_steps=2,
                         check_val_every_n_epoch=params_module.PARAMS["val_interval"],
                         profiler=profiler,
                        #  detect_anomaly=True
                         )

    with open(params_path) as f:
        write_params = f.read()
    trainer.logger.experiment.add_text(params_path, write_params)

    trainer.fit(lit_module, train_loader, [val_loader_devin, val_loader_evalin])            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get all command line arguments.')
    parser.add_argument('params', type=str, help='Path to parameters py file')
    args = parser.parse_args()

    # load params file as module
    spec = importlib.util.spec_from_file_location("params", args.params)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)

    main(args.params, params)
