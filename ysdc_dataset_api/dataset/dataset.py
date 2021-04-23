import json
import os
import random

import numpy as np
import torch
from google.protobuf.internal.decoder import _DecodeVarint32

from ysdc_dataset_api.proto import Scene, get_tags_from_request, proto_to_dict
from ysdc_dataset_api.rendering import FeatureRenderer
from ysdc_dataset_api.utils import (
    get_track_for_transform,
    get_track_to_fm_transform,
    request_is_valid,
)


N_SCENES_PER_FILE = 5000
N_SCENES_PER_SUBDIR = 1000


class MotionPredictionDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            dataset_path,
            scene_tags_fpath,
            renderer_config=None,
            scene_tags_filter=None,
            trajectory_tags_filter=None,
        ):
        super(MotionPredictionDataset, self).__init__()
        self._renderer = FeatureRenderer(renderer_config)

        self._scene_tags_filter = _callable_or_trivial_filter(scene_tags_filter)
        self._trajectory_tags_filter = _callable_or_trivial_filter(trajectory_tags_filter)

        self._scene_file_paths = self._filter_paths(
            get_file_paths(dataset_path), scene_tags_fpath)

    @property
    def num_scenes(self):
        return len(self._scene_file_paths)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            file_paths = self._scene_file_paths
        else:
            file_paths = self._split_filepaths_by_worker(worker_info.id, worker_info.num_workers)

        def data_gen(_file_paths):
            for fpath in file_paths:
                scene = _read_scene_from_file(fpath)
                for request in scene.prediction_requests:
                    if not request_is_valid(scene, request):
                        continue
                    trajectory_tags = get_tags_from_request(request)
                    if not self._trajectory_tags_filter(trajectory_tags):
                        continue
                    track = get_track_for_transform(scene, request.track_id)
                    track_to_fm_transform = get_track_to_fm_transform(track)
                    feature_maps = self._renderer.render_features(scene, track_to_fm_transform)
                    gt_trajectory = transform_points(
                        get_gt_trajectory(scene, request.track_id), transform)
                    yield {
                        'feature_maps': feature_maps,
                        'gt_trajectory': gt_trajectory,
                    }

        return data_gen(file_paths)

    def _split_filepaths_by_worker(self, worker_id, num_workers):
        n_scenes_per_worker = self.num_scenes // num_workers
        split = list(range(0, self.num_scenes, n_scenes_per_worker))
        start = split[worker_id]
        stop = split[worker_id + 1]
        if worker_id == num_workers - 1:
            stop = self.num_scenes
        return self._scene_file_paths[start:stop]

    def _callable_or_lambda_true(self, f):
        if f is None:
            return lambda x: True
        if not callable(f):
            raise ValueError('Expected callable, got {}'.format(type(f)))
        return f

    def _filter_paths(self, file_paths, scene_tags_fpath):
        valid_indices = []
        with open(scene_tags_fpath, 'r') as f:
            for i, line in enumerate(f):
                tags = json.loads(line.strip())
                if self._scene_tags_filter(tags):
                    valid_indices.append(i)
        return [file_paths[i] for i in valid_indices]


class MotionPredictionDatasetV2(torch.utils.data.Dataset):
    def __init__(self, dataset_path, scene_tags_filter=None, trajectory_tags_filter=None):
        super(MotionPredictionDatasetV2, self).__init__()

        self._scene_tags_filter = _callable_or_trivial_filter(scene_tags_filter)
        self._trajectory_tags_filter = _callable_or_trivial_filter(trajectory_tags_filter)

        self._dataset_files = self._filter_scenes(get_file_paths(dataset_path))

    def __getitem__(self, idx):
        scene = self._read_scene_from_file(self._dataset_files[idx])
        return {
            'pedestrians': np.random.rand(5, 25, 2),
            'vehicles': np.random.rand(5, 25, 2),
            'gt_vehicles': np.random.rand(5, 25, 2),
        }

    def __len__(self):
        # This is broken due to more than one actor in request per scene
        return len(self._dataset_files)

    def _read_scene_by_idx(self, scene_idx):
        dir_num = scene_idx // N_SCENES_PER_SUBDIR
        fpath = os.path.join(self._dataset_path, f'{dir_num:03d}', f'{scene_idx:06d}.pb')
        return _read_scene_from_file(fpath)

    def _filter_scenes(self, indices):
        return indices


def get_file_paths(dataset_path):
    sub_dirs = sorted([
        os.path.join(dataset_path, d) for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])
    res = []
    for d in sub_dirs:
        file_names = sorted(os.listdir(os.path.join(dataset_path, d)))
        res += [os.path.join(d, fname) for fname in file_names]
    return res


def _read_scene_from_file(filepath):
    with open(filepath, 'rb') as f:
        scene_serialized = f.read()
    scene = Scene()
    scene.ParseFromString(scene_serialized)
    return scene


def _callable_or_trivial_filter(f):
    if f is None:
        return _trivial_filter
    if not callable(f):
        raise ValueError('Expected callable, got {}'.format(type(f)))
    return f


def _trivial_filter(x):
    return x


def _dataset_file_iterator(filepath, start_ind=0, stop_ind=-1):
    with open(filepath, 'rb') as f:
        file_content = f.read()
    sep_pos = 0
    scene_ind = 0
    while sep_pos < len(file_content):
        msg_len, new_sep_pos = _DecodeVarint32(file_content, sep_pos)
        sep_pos = new_sep_pos
        encoded_message = file_content[sep_pos:sep_pos + msg_len]
        sep_pos += msg_len
        if scene_ind < start_ind:
            scene_ind += 1
            continue
        if stop_ind > 0 and scene_ind == stop_ind:
            break
        scene = Scene.FromString(encoded_message)
        scene_ind += 1
        yield scene
