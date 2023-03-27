
import SimpleITK
import numpy as np
import torch
from scipy import ndimage
# from monai.networks.nets import UNet
from x_unet import XUnet
from monai.inferers import sliding_window_inference
from uncertainty import ensemble_uncertainties_classification
from pathlib import Path
from collections import OrderedDict

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        # print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def remove_connected_components(segmentation, l_min=9):
    """
    Remove all lesions with less or equal amount of voxels than `l_min` from a 
    binary segmentation mask `segmentation`.
    Args:
      segmentation: `numpy.ndarray` of shape [H, W, D], with a binary lesions segmentation mask.
      l_min:  `int`, minimal amount of voxels in a lesion.
    Returns:
      Binary lesion segmentation mask (`numpy.ndarray` of shape [H, W, D])
      only with connected components that have more than `l_min` voxels.
    """
    labeled_seg, num_labels = ndimage.label(segmentation)
    label_list = np.unique(labeled_seg)
    num_elements_by_lesion = ndimage.labeled_comprehension(segmentation, labeled_seg, label_list, np.sum, float, 0)

    seg2 = np.zeros_like(segmentation)
    for i_el, n_el in enumerate(num_elements_by_lesion):
        if n_el > l_min:
            current_voxels = np.stack(np.where(labeled_seg == i_el), axis=1)
            seg2[current_voxels[:, 0],
                 current_voxels[:, 1],
                 current_voxels[:, 2]] = 1
    return seg2
    

class XUnet_Algorithm(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    # UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        print(torch.__version__)
        output_path = Path("/output/images/")
        if not output_path.exists():
            output_path.mkdir()

        # self._input_path = Path("/input/images/brain-mri/")
        # self._input_path = Path("./input/")
        self._segmentation_output_path = Path("/output/images/white-matter-multiple-sclerosis-lesion-segmentation/")
        self._uncertainty_output_path = Path("/output/images/white-matter-multiple-sclerosis-lesion-uncertainty-map/")

        self.device = get_default_device()

        model = XUnet(dim = 64,
                      frame_kernel_size = 3,
                      channels = 1,
                      out_dim=2,
                      attn_dim_head = 32,
                      attn_heads = 8,
                      dim_mults = (1, 2, 4, 8),
                      num_blocks_per_stage = (2, 2, 2, 2),
                      num_self_attn_per_stage = (0, 0, 0, 1),
                      nested_unet_depths = (5, 4, 2, 1),
                      consolidate_upsample_fmaps = True,
                      weight_standardize = False)
        model = torch.compile(model)
        checkpoint = torch.load('./model.ckpt', map_location='cpu')
        checkpoint = OrderedDict({k.replace('model.', '', 1):v for k,v in checkpoint['state_dict'].items()})
        model.load_state_dict(checkpoint)

        if self.device == torch.device('cuda'):
            model.half()

        model.to(self.device)
        model.eval()

        self.model = model
        self.act = torch.nn.Softmax(dim=1)
        self.th = 0.35
        self.roi_size = (64, 64, 64)
        self.sw_batch_size = 10


    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Segment nodule candidates
        segmented_map, uncertainty_map = self.predict(input_image=input_image)

        # Write resulting segmentation to output location
        segmentation_path = self._segmentation_output_path / input_image_file_path.name
        if not self._segmentation_output_path.exists():
            self._segmentation_output_path.mkdir()
        SimpleITK.WriteImage(segmented_map, str(segmentation_path), True)

        # Write resulting uncertainty map to output location
        uncertainty_path = self._uncertainty_output_path / input_image_file_path.name
        if not self._uncertainty_output_path.exists():
            self._uncertainty_output_path.mkdir()
        SimpleITK.WriteImage(uncertainty_map, str(uncertainty_path), True)

        # Write segmentation file path to result.json for this case
        return {
            "segmentation": [
                dict(type="metaio_image", filename=segmentation_path.name)
            ],
            "uncertainty": [
                dict(type="metaio_image", filename=uncertainty_path.name)
            ],
            "inputs": [
                dict(type="metaio_image", filename=input_image_file_path.name)
            ],
            "error_messages": [],
        }


    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
        image = SimpleITK.GetArrayFromImage(input_image)
        image = np.transpose(np.array(image))

        # normalize values
        mina = image.min()
        maxa = image.max()
        image = (image - mina) / (maxa - mina)

        with torch.no_grad():
            image = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(image), axis=0), axis=0)

            if self.device == torch.device('cuda'):
                image = image.half()

            outputs = sliding_window_inference(image.to(self.device), self.roi_size, self.sw_batch_size, self.model, mode='gaussian')
            outputs = self.act(outputs).cpu().numpy().astype(np.float32)
            outputs = np.squeeze(outputs[0,1])

        seg = outputs.copy()
        seg[seg>self.th] = 1
        seg[seg<=self.th] = 0
        seg = np.squeeze(seg)
        seg = remove_connected_components(seg)

        uncs_map = ensemble_uncertainties_classification(np.concatenate(
                    (np.expand_dims(outputs, axis=(0,-1)),
                     np.expand_dims(1. - outputs, axis=(0,-1))),
                    axis=-1))['entropy_of_expected']

        out_seg = SimpleITK.GetImageFromArray(seg)
        out_unc = SimpleITK.GetImageFromArray(uncs_map)
        return out_seg, out_unc


if __name__ == "__main__":
    XUnet_Algorithm().process()
