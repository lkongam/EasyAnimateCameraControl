import csv
import gc
import io
import json
import math
import os
import random
from contextlib import contextmanager
from threading import Thread

import albumentations
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from func_timeout import FunctionTimedOut, func_timeout
from packaging import version as pver
from PIL import Image
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset
from typing import List, Dict, Any, Optional, Union, Tuple

VIDEO_READER_TIMEOUT = 20


def get_random_mask(shape):
    f, c, h, w = shape

    if f != 1:
        mask_index = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.05, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05])
    else:
        mask_index = np.random.choice([0, 1], p=[0.2, 0.8])
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)

    if mask_index == 0:
        center_x = torch.randint(0, w, (1,)).item()
        center_y = torch.randint(0, h, (1,)).item()
        block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
        block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

        start_x = max(center_x - block_size_x // 2, 0)
        end_x = min(center_x + block_size_x // 2, w)
        start_y = max(center_y - block_size_y // 2, 0)
        end_y = min(center_y + block_size_y // 2, h)
        mask[:, :, start_y:end_y, start_x:end_x] = 1
    elif mask_index == 1:
        mask[:, :, :, :] = 1
    elif mask_index == 2:
        mask_frame_index = np.random.randint(1, 5)
        mask[mask_frame_index:, :, :, :] = 1
    elif mask_index == 3:
        mask_frame_index = np.random.randint(1, 5)
        mask[mask_frame_index:-mask_frame_index, :, :, :] = 1
    elif mask_index == 4:
        center_x = torch.randint(0, w, (1,)).item()
        center_y = torch.randint(0, h, (1,)).item()
        block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
        block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

        start_x = max(center_x - block_size_x // 2, 0)
        end_x = min(center_x + block_size_x // 2, w)
        start_y = max(center_y - block_size_y // 2, 0)
        end_y = min(center_y + block_size_y // 2, h)

        mask_frame_before = np.random.randint(0, f // 2)
        mask_frame_after = np.random.randint(f // 2, f)
        mask[mask_frame_before:mask_frame_after, :, start_y:end_y, start_x:end_x] = 1
    elif mask_index == 5:
        mask = torch.randint(0, 2, (f, 1, h, w), dtype=torch.uint8)
    elif mask_index == 6:
        num_frames_to_mask = random.randint(1, max(f // 2, 1))
        frames_to_mask = random.sample(range(f), num_frames_to_mask)

        for i in frames_to_mask:
            block_height = random.randint(1, h // 4)
            block_width = random.randint(1, w // 4)
            top_left_y = random.randint(0, h - block_height)
            top_left_x = random.randint(0, w - block_width)
            mask[i, 0, top_left_y : top_left_y + block_height, top_left_x : top_left_x + block_width] = 1
    elif mask_index == 7:
        center_x = torch.randint(0, w, (1,)).item()
        center_y = torch.randint(0, h, (1,)).item()
        a = torch.randint(min(w, h) // 8, min(w, h) // 4, (1,)).item()  # 长半轴
        b = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()  # 短半轴

        for i in range(h):
            for j in range(w):
                if ((i - center_y) ** 2) / (b**2) + ((j - center_x) ** 2) / (a**2) < 1:
                    mask[:, :, i, j] = 1
    elif mask_index == 8:
        center_x = torch.randint(0, w, (1,)).item()
        center_y = torch.randint(0, h, (1,)).item()
        radius = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()
        for i in range(h):
            for j in range(w):
                if (i - center_y) ** 2 + (j - center_x) ** 2 < radius**2:
                    mask[:, :, i, j] = 1
    elif mask_index == 9:
        for idx in range(f):
            if np.random.rand() > 0.5:
                mask[idx, :, :, :] = 1
    else:
        raise ValueError(f"The mask_index {mask_index} is not define")
    return mask


class ImageVideoSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self, sampler: Sampler, dataset: Dataset, batch_size: int, drop_last: bool = False) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, ' f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, ' f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.bucket = {'image': [], 'video': []}

    def __iter__(self):
        for idx in self.sampler:
            content_type = self.dataset.dataset[idx].get('type', 'image')
            self.bucket[content_type].append(idx)

            # yield a batch of indices in the same aspect ratio group
            if len(self.bucket['video']) == self.batch_size:
                bucket = self.bucket['video']
                yield bucket[:]
                del bucket[:]
            elif len(self.bucket['image']) == self.batch_size:
                bucket = self.bucket['image']
                yield bucket[:]
                del bucket[:]


class ALLDatasetsSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self, sampler: Sampler, dataset: Dataset, batch_size: int, drop_last: bool = False) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, ' f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, ' f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.bucket = {'objaverse': [], 'kubric': [], 'realestate': []}

    def __iter__(self):
        for idx in self.sampler:
            try:
                content_type = self.dataset[idx].get('type', 'objaverse')
                self.bucket[content_type].append(idx)

                # yield a batch of indices in the same aspect ratio group
                if len(self.bucket['objaverse']) == self.batch_size:
                    bucket = self.bucket['objaverse']
                    yield bucket[:]
                    del bucket[:]
                elif len(self.bucket['realestate']) == self.batch_size:
                    bucket = self.bucket['realestate']
                    yield bucket[:]
                    del bucket[:]
                elif len(self.bucket['kubric']) == self.batch_size:
                    bucket = self.bucket['kubric']
                    yield bucket[:]
                    del bucket[:]
            except Exception as e:
                # 可选：记录错误信息以便调试
                # print(f"ALLDatasetsSampler __iter__ Error processing index {idx}: {e}")
                continue  # 跳过当前迭代，继续下一个


@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()


def get_video_reader_batch(video_reader, batch_index):
    frames = video_reader.get_batch(batch_index).asnumpy()
    return frames


def resize_frame(frame, target_short_side):
    h, w, _ = frame.shape
    if h < w:
        if target_short_side > h:
            return frame
        new_h = target_short_side
        new_w = int(target_short_side * w / h)
    else:
        if target_short_side > w:
            return frame
        new_w = target_short_side
        new_h = int(target_short_side * h / w)

    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame


class ImageVideoDataset(Dataset):
    def __init__(
        self,
        ann_path,
        data_root=None,
        video_sample_size='384x672',
        video_sample_stride=4,
        video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=-1,
        enable_bucket=False,
        video_length_drop_start=0.1,
        video_length_drop_end=0.9,
        enable_inpaint=False,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))

        self.data_root = data_root

        # It's used to balance num of images and videos.
        self.dataset = []
        for data in dataset:
            if data.get('type', 'image') != 'video':
                self.dataset.append(data)
        if video_repeat > 0:
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        # Image params
        self.image_sample_size = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.image_sample_size)),
                transforms.CenterCrop(self.image_sample_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]

        if data_info.get('type', 'image') == 'video':
            video_id, text = data_info['file_path'], data_info['text']

            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)

            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args)
                    resized_frames = []
                    for i in range(len(pixel_values)):
                        frame = pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.0
                    del video_reader
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    pixel_values = self.video_transforms(pixel_values)

                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''
            return pixel_values, text, 'video'
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)
            if random.random() < self.text_drop_ratio:
                text = ''
            return image, text, 'image'

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, name, data_type = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx

                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length - 1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.ones_like(pixel_values) * -1 * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

            ref_pixel_values = sample["pixel_values"][0].unsqueeze(0)
            if (mask == 1).all():
                ref_pixel_values = torch.ones_like(ref_pixel_values) * -1
            sample["ref_pixel_values"] = ref_pixel_values

        return sample


class ImageVideoControlDataset(Dataset):
    def __init__(
        self,
        ann_path,
        data_root=None,
        video_sample_size=512,
        video_sample_stride=4,
        video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=-1,
        enable_bucket=False,
        video_length_drop_start=0.1,
        video_length_drop_end=0.9,
        enable_inpaint=False,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))

        self.data_root = data_root

        # It's used to balance num of images and videos.
        self.dataset = []
        for data in dataset:
            if data.get('type', 'image') != 'video':
                self.dataset.append(data)
        if video_repeat > 0:
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        # Image params
        self.image_sample_size = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.image_sample_size)),
                transforms.CenterCrop(self.image_sample_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        video_id, text = data_info['file_path'], data_info['text']

        if data_info.get('type', 'image') == 'video':
            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)

            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args)
                    resized_frames = []
                    for i in range(len(pixel_values)):
                        frame = pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.0
                    del video_reader
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    pixel_values = self.video_transforms(pixel_values)

                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''

            control_video_id = data_info['control_file_path']

            if self.data_root is None:
                control_video_id = control_video_id
            else:
                control_video_id = os.path.join(self.data_root, control_video_id)

            with VideoReader_contextmanager(control_video_id, num_threads=2) as control_video_reader:
                try:
                    sample_args = (control_video_reader, batch_index)
                    control_pixel_values = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args)
                    resized_frames = []
                    for i in range(len(control_pixel_values)):
                        frame = control_pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    control_pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    control_pixel_values = torch.from_numpy(control_pixel_values).permute(0, 3, 1, 2).contiguous()
                    control_pixel_values = control_pixel_values / 255.0
                    del control_video_reader
                else:
                    control_pixel_values = control_pixel_values

                if not self.enable_bucket:
                    control_pixel_values = self.video_transforms(control_pixel_values)
            return pixel_values, control_pixel_values, text, "video"
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)

            if random.random() < self.text_drop_ratio:
                text = ''

            control_image_id = data_info['control_file_path']

            if self.data_root is None:
                control_image_id = control_image_id
            else:
                control_image_id = os.path.join(self.data_root, control_image_id)

            control_image = Image.open(control_image_id).convert('RGB')
            if not self.enable_bucket:
                control_image = self.image_transforms(control_image).unsqueeze(0)
            else:
                control_image = np.expand_dims(np.array(control_image), 0)
            return image, control_image, text, 'image'

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, control_pixel_values, name, data_type = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["control_pixel_values"] = control_pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx

                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length - 1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.ones_like(pixel_values) * -1 * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

            ref_pixel_values = sample["pixel_values"][0].unsqueeze(0)
            if (mask == 1).all():
                ref_pixel_values = torch.ones_like(ref_pixel_values) * -1
            sample["ref_pixel_values"] = ref_pixel_values

        return sample


def get_video_from_dir(video_dir):
    cap = cv2.VideoCapture(video_dir)

    if not cap.isOpened():
        print(f"Cannot Open the Video File: {video_dir}")
        return []

    whole_video = []
    frame_count = 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        whole_video.append(frame_pil)
        frame_count += 1

    cap.release()

    return whole_video, frame_height, frame_width


def get_camera_from_dir(pose_file_dir):
    whole_camera_para = []

    try:
        with open(pose_file_dir, 'r', encoding='utf-8') as file:
            # 读取所有行
            lines = file.readlines()

            # 确保文件至少有两行
            if len(lines) < 2:
                print("文件内容不足两行，无法读取数据。")
                return whole_camera_para

            # 跳过第一行，从第二行开始处理
            for idx, line in enumerate(lines[1:], start=2):
                # 去除首尾空白字符并按空格分割
                parts = line.strip().split()

                # 检查每行是否有19个数字
                if len(parts) != 19:
                    print(f"警告：第 {idx} 行的数字数量不是19，跳过该行。")
                    continue

                try:
                    # 将字符串转换为浮点数
                    numbers = [float(part) for part in parts]
                    whole_camera_para.append(numbers)
                except ValueError as ve:
                    print(f"警告：第 {idx} 行包含非数字字符，跳过该行。错误详情: {ve}")
                    continue

    except FileNotFoundError:
        print(f"错误：文件 '{pose_file_dir}' 未找到。")
    except Exception as e:
        print(f"发生错误：{e}")

    return whole_camera_para


class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        try:
            self.c2w_mat = np.linalg.inv(w2c_mat_4x4)
        except:
            self.c2w_mat = np.linalg.pinv(w2c_mat_4x4)


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype), torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype))
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1
    fx += 1e-10
    fy += 1e-10
    zs = torch.ones_like(i)  # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, HW, 3
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.linalg.cross(rays_o, rays_d)  # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


def get_relative_pose(cam_params):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]

    target_cam_c2w = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def compute_plucker(cam_params, video_sample_size, ori_h, ori_w):

    cam_params = [Camera(cam_param) for cam_param in cam_params]

    ori_wh_ratio = ori_w / ori_h
    sample_wh_ratio = video_sample_size[1] / video_sample_size[0]
    if ori_wh_ratio > sample_wh_ratio:  # rescale fx
        resized_ori_w = video_sample_size[0] * ori_wh_ratio
        for cam_param in cam_params:
            cam_param.fx = resized_ori_w * cam_param.fx / video_sample_size[1]
    else:  # rescale fy
        resized_ori_h = video_sample_size[1] / ori_wh_ratio
        for cam_param in cam_params:
            cam_param.fy = resized_ori_h * cam_param.fy / video_sample_size[0]

    intrinsics = np.asarray(
        [
            [cam_param.fx * video_sample_size[1], cam_param.fy * video_sample_size[0], cam_param.cx * video_sample_size[1], cam_param.cy * video_sample_size[0]]
            for cam_param in cam_params
        ],
        dtype=np.float32,
    )

    intrinsics = torch.as_tensor(intrinsics)[None]  # [1, n_frame, 4]
    c2w_poses = get_relative_pose(cam_params)
    c2w = torch.as_tensor(c2w_poses)[None]  # [1, n_frame, 4, 4]
    plucker_embedding = ray_condition(intrinsics, c2w, video_sample_size[0], video_sample_size[1], device='cpu')[0].permute(0, 3, 1, 2).contiguous()

    return plucker_embedding


def get_plucker_embedding(whole_camera_para, batch_index_input, batch_index_output, video_sample_size, ori_h, ori_w):
    input_camera_para = [whole_camera_para[i] for i in batch_index_input]
    output_camera_para = [whole_camera_para[i] for i in batch_index_output]

    plucker_embedding_input = compute_plucker(input_camera_para, video_sample_size, ori_h, ori_w)
    plucker_embedding_output = compute_plucker(output_camera_para, video_sample_size, ori_h, ori_w)

    return plucker_embedding_input, plucker_embedding_output


def get_pixel_value(whole_video, batch_index_input, batch_index_output, video_transforms):
    # 提取输入和输出帧
    input_frames = [whole_video[i] for i in batch_index_input]
    output_frames = [whole_video[i] for i in batch_index_output]

    # 定义 ToTensor 转换，如果 self.video_transforms 中已经包含了 ToTensor，则可以省略
    to_tensor = transforms.ToTensor()

    # 处理输入帧
    input_tensors = []
    for frame in input_frames:
        tensor = to_tensor(frame)  # 转换为 [C, H, W] 并归一化到 [0, 1]
        input_tensors.append(tensor)
    pixel_values_input = torch.stack(input_tensors, dim=0)  # 形状 [n, C, H, W]

    # 应用额外的转换（如果有的话）
    if video_transforms:
        pixel_values_input = video_transforms(pixel_values_input)

    # 处理输出帧
    output_tensors = []
    for frame in output_frames:
        tensor = to_tensor(frame)  # 转换为 [C, H, W] 并归一化到 [0, 1]
        output_tensors.append(tensor)
    pixel_values_output = torch.stack(output_tensors, dim=0)  # 形状 [n, C, H, W]

    # 应用额外的转换（如果有的话）
    if video_transforms:
        pixel_values_output = video_transforms(pixel_values_output)

    return pixel_values_input, pixel_values_output


def get_empty_plucker_embedding(pixel_values_input, direction_number=6):
    shape = list(pixel_values_input.shape)
    shape[1] = direction_number
    shape = tuple(shape)
    dtype = pixel_values_input.dtype
    device = pixel_values_input.device
    plucker_in = torch.zeros(shape, device=device, dtype=dtype)
    plucker_out = torch.zeros(shape, device=device, dtype=dtype)

    return plucker_in, plucker_out


class GenXD(Dataset):
    def __init__(
        self,
        GenXD_dataset_list,
        data_root,
        video_sample_stride,
        video_sample_n_frames,
        video_sample_size,
        enable_inpaint=False,
    ):
        self.GenXD_dataset_list = GenXD_dataset_list
        self.data_root = data_root
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = video_sample_size
        self.enable_inpaint = enable_inpaint
        self.length = len(self.GenXD_dataset_list)

        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def get_batch_index(self, length_of_video):
        if length_of_video >= self.video_sample_n_frames:

            def compute_stride(num_frames, desired_stride, num_samples):
                max_possible_stride = num_frames // num_samples
                if max_possible_stride == 0:
                    return 1  # 最小步长为1
                return min(desired_stride, max_possible_stride)

            # 计算输入和输出部分的步长
            stride = compute_stride(length_of_video, self.video_sample_stride, self.video_sample_n_frames)

            def get_indices(start, num_frames, stride, total_length):
                indices = [start + i * stride for i in range(num_frames)]
                # 确保索引不超过总长度
                if indices[-1] >= total_length:
                    # 如果最后一个索引超出范围，调整起始点
                    start = total_length - (num_frames * stride)
                    start = max(start, 0)
                    indices = [start + i * stride for i in range(num_frames)]
                return indices

            # 获取输入部分的帧索引
            batch_index_output = get_indices(start=0, num_frames=self.video_sample_n_frames, stride=stride, total_length=length_of_video)
        else:
            # 当视频长度小于样本帧数时，均匀重复帧
            interval = length_of_video / self.video_sample_n_frames
            batch_index_output = [min(int(i * interval), length_of_video - 1) for i in range(self.video_sample_n_frames)]

        batch_index_input = [batch_index_output[0]] * self.video_sample_n_frames

        return batch_index_input, batch_index_output

    def get_batch(self, data_info):
        data_type = data_info['type']

        video_id, pose_file = data_info['video_file_path'], data_info['camera_file_path']

        if self.data_root is None:
            video_dir = video_id
            pose_file_dir = pose_file
        else:
            video_dir = os.path.join(self.data_root, video_id)
            pose_file_dir = os.path.join(self.data_root, pose_file)

        whole_video, ori_h, ori_w = get_video_from_dir(video_dir)
        whole_camera_para = get_camera_from_dir(pose_file_dir)

        # print(len(whole_video))
        # print(len(whole_camera_para))

        assert len(whole_video) == len(whole_camera_para), "the length of video must be euqal with that of camera parameter."

        batch_index_input, batch_index_output = self.get_batch_index(len(whole_video))
        plucker_embedding_input, plucker_embedding_output = get_plucker_embedding(
            whole_camera_para,
            batch_index_input,
            batch_index_output,
            self.video_sample_size,
            ori_h,
            ori_w,
        )
        pixel_values_input, pixel_values_output = get_pixel_value(whole_video, batch_index_input, batch_index_output, self.video_transforms)

        return pixel_values_input, pixel_values_output, plucker_embedding_input, plucker_embedding_output, data_type

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.GenXD_dataset_list[idx]
        sample = {}

        pixel_values_input, pixel_values_output, plucker_embedding_input, plucker_embedding_output, data_type = self.get_batch(data_info)
        sample["pixel_values_input"] = pixel_values_input
        sample["pixel_values_output"] = pixel_values_output
        sample["plucker_embedding_input"] = plucker_embedding_input
        sample["plucker_embedding_output"] = plucker_embedding_output
        sample['text'] = data_info['text']
        sample["data_type"] = data_type
        sample["idx"] = idx

        if self.enable_inpaint:
            mask = get_random_mask(pixel_values_output.size())
            mask_pixel_values = pixel_values_output * (1 - mask) + torch.ones_like(pixel_values_output) * -1 * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values_input"].permute(0, 2, 3, 1).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

            # ref_pixel_values = sample["pixel_values_input"][0].unsqueeze(0)
            # if (mask == 1).all():
            #     ref_pixel_values = torch.ones_like(ref_pixel_values) * -1
            # sample["ref_pixel_values"] = ref_pixel_values

        return sample


class KubricDataset(Dataset):
    def __init__(
        self,
        objaverse_dataset_list,
        data_root,
        video_sample_stride,
        video_sample_n_frames,
        video_sample_size,
        enable_inpaint=False,
    ):
        self.objaverse_dataset_list = objaverse_dataset_list
        self.data_root = data_root
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = video_sample_size
        self.enable_inpaint = enable_inpaint
        self.length = len(self.objaverse_dataset_list)

        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def get_batch_index(self, length_of_video):
        mid = length_of_video // 2
        front_half = mid
        back_half = length_of_video - mid

        def compute_stride(num_frames, desired_stride, num_samples):
            max_possible_stride = num_frames // num_samples
            if max_possible_stride == 0:
                return 1  # 最小步长为1
            return min(desired_stride, max_possible_stride)

        # 计算输入和输出部分的步长
        stride_input = compute_stride(front_half, self.video_sample_stride, self.video_sample_n_frames)
        stride_output = compute_stride(back_half, self.video_sample_stride, self.video_sample_n_frames)

        def get_indices(start, num_frames, stride, total_length):
            indices = [start + i * stride for i in range(num_frames)]
            # 确保索引不超过总长度
            if indices[-1] >= total_length:
                # 如果最后一个索引超出范围，调整起始点
                start = total_length - (num_frames * stride)
                start = max(start, 0)
                indices = [start + i * stride for i in range(num_frames)]
            return indices

        # 获取输入部分的帧索引
        batch_index_input = get_indices(start=0, num_frames=self.video_sample_n_frames, stride=stride_input, total_length=front_half)

        # 获取输出部分的帧索引
        batch_index_output = get_indices(start=mid, num_frames=self.video_sample_n_frames, stride=stride_output, total_length=length_of_video)

        return batch_index_input, batch_index_output

    def get_batch(self, data_info):
        data_type = data_info['type']

        video_id, pose_file = data_info['video_file_path'], data_info['camera_file_path']

        if self.data_root is None:
            video_dir = video_id
            pose_file_dir = pose_file
        else:
            video_dir = os.path.join(self.data_root, video_id)
            pose_file_dir = os.path.join(self.data_root, pose_file)

        whole_video, ori_h, ori_w = get_video_from_dir(video_dir)
        whole_camera_para = get_camera_from_dir(pose_file_dir)

        # print(len(whole_video))
        # print(len(whole_camera_para))

        assert len(whole_video) == len(whole_camera_para), "the length of video must be euqal with that of camera parameter."

        batch_index_input, batch_index_output = self.get_batch_index(len(whole_video))
        plucker_embedding_input, plucker_embedding_output = get_plucker_embedding(
            whole_camera_para,
            batch_index_input,
            batch_index_output,
            self.video_sample_size,
            ori_h,
            ori_w,
        )
        pixel_values_input, pixel_values_output = get_pixel_value(whole_video, batch_index_input, batch_index_output, self.video_transforms)

        return pixel_values_input, pixel_values_output, plucker_embedding_input, plucker_embedding_output, data_type

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.objaverse_dataset_list[idx]
        sample = {}

        pixel_values_input, pixel_values_output, plucker_embedding_input, plucker_embedding_output, data_type = self.get_batch(data_info)
        sample["pixel_values_input"] = pixel_values_input
        sample["pixel_values_output"] = pixel_values_output
        sample["plucker_embedding_input"] = plucker_embedding_input
        sample["plucker_embedding_output"] = plucker_embedding_output
        sample['text'] = data_info['text']
        sample["data_type"] = data_type
        sample["idx"] = idx

        if self.enable_inpaint:
            mask = get_random_mask(pixel_values_output.size())
            mask_pixel_values = pixel_values_output * (1 - mask) + torch.ones_like(pixel_values_output) * -1 * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values_input"].permute(0, 2, 3, 1).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

            # ref_pixel_values = sample["pixel_values_input"][0].unsqueeze(0)
            # if (mask == 1).all():
            #     ref_pixel_values = torch.ones_like(ref_pixel_values) * -1
            # sample["ref_pixel_values"] = ref_pixel_values

        return sample


class ObjaverseDataset(Dataset):
    def __init__(
        self,
        objaverse_dataset_list,
        data_root,
        video_sample_stride,
        video_sample_n_frames,
        video_sample_size,
        enable_inpaint=False,
    ):
        self.objaverse_dataset_list = objaverse_dataset_list
        self.data_root = data_root
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = video_sample_size
        self.enable_inpaint = enable_inpaint
        self.length = len(self.objaverse_dataset_list)

        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def get_batch_index(self, length_of_video):
        mid = length_of_video // 2
        front_half = mid
        back_half = length_of_video - mid

        def compute_stride(num_frames, desired_stride, num_samples):
            max_possible_stride = num_frames // num_samples
            if max_possible_stride == 0:
                return 1  # 最小步长为1
            return min(desired_stride, max_possible_stride)

        # 计算输入和输出部分的步长
        stride_input = compute_stride(front_half, self.video_sample_stride, self.video_sample_n_frames)
        stride_output = compute_stride(back_half, self.video_sample_stride, self.video_sample_n_frames)

        def get_indices(start, num_frames, stride, total_length):
            indices = [start + i * stride for i in range(num_frames)]
            # 确保索引不超过总长度
            if indices[-1] >= total_length:
                # 如果最后一个索引超出范围，调整起始点
                start = total_length - (num_frames * stride)
                start = max(start, 0)
                indices = [start + i * stride for i in range(num_frames)]
            return indices

        # 获取输入部分的帧索引
        batch_index_input = get_indices(start=0, num_frames=self.video_sample_n_frames, stride=stride_input, total_length=front_half)

        # 获取输出部分的帧索引
        batch_index_output = get_indices(start=mid, num_frames=self.video_sample_n_frames, stride=stride_output, total_length=length_of_video)

        return batch_index_input, batch_index_output

    def get_batch(self, data_info):
        data_type = data_info['type']

        video_id, pose_file = data_info['video_file_path'], data_info['camera_file_path']

        if self.data_root is None:
            video_dir = video_id
            pose_file_dir = pose_file
        else:
            video_dir = os.path.join(self.data_root, video_id)
            pose_file_dir = os.path.join(self.data_root, pose_file)

        whole_video, ori_h, ori_w = get_video_from_dir(video_dir)
        whole_camera_para = get_camera_from_dir(pose_file_dir)

        # print(len(whole_video))
        # print(len(whole_camera_para))

        assert len(whole_video) == len(whole_camera_para), "the length of video must be euqal with that of camera parameter."

        batch_index_input, batch_index_output = self.get_batch_index(len(whole_video))
        plucker_embedding_input, plucker_embedding_output = get_plucker_embedding(
            whole_camera_para,
            batch_index_input,
            batch_index_output,
            self.video_sample_size,
            ori_h,
            ori_w,
        )
        pixel_values_input, pixel_values_output = get_pixel_value(whole_video, batch_index_input, batch_index_output, self.video_transforms)

        return pixel_values_input, pixel_values_output, plucker_embedding_input, plucker_embedding_output, data_type

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.objaverse_dataset_list[idx]
        sample = {}

        pixel_values_input, pixel_values_output, plucker_embedding_input, plucker_embedding_output, data_type = self.get_batch(data_info)
        sample["pixel_values_input"] = pixel_values_input
        sample["pixel_values_output"] = pixel_values_output
        sample["plucker_embedding_input"] = plucker_embedding_input
        sample["plucker_embedding_output"] = plucker_embedding_output
        sample['text'] = data_info['text']
        sample["data_type"] = data_type
        sample["idx"] = idx

        if self.enable_inpaint:
            mask = get_random_mask(pixel_values_output.size())
            mask_pixel_values = pixel_values_output * (1 - mask) + torch.ones_like(pixel_values_output) * -1 * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values_input"].permute(0, 2, 3, 1).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

            # ref_pixel_values = sample["pixel_values_input"][0].unsqueeze(0)
            # if (mask == 1).all():
            #     ref_pixel_values = torch.ones_like(ref_pixel_values) * -1
            # sample["ref_pixel_values"] = ref_pixel_values

        return sample


class RealEstateDataset(Dataset):
    def __init__(
        self,
        realestate_dataset_list,
        data_root,
        video_sample_stride,
        video_sample_n_frames,
        video_sample_size,
        enable_inpaint=False,
    ):
        self.realestate_dataset_list = realestate_dataset_list
        self.data_root = data_root
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = video_sample_size
        self.enable_inpaint = enable_inpaint
        self.length = len(self.realestate_dataset_list)

        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def get_batch_index(self, length_of_video):
        if length_of_video >= self.video_sample_n_frames:

            def compute_stride(num_frames, desired_stride, num_samples):
                max_possible_stride = num_frames // num_samples
                if max_possible_stride == 0:
                    return 1  # 最小步长为1
                return min(desired_stride, max_possible_stride)

            # 计算输入和输出部分的步长
            stride = compute_stride(length_of_video, self.video_sample_stride, self.video_sample_n_frames)

            def get_indices(start, num_frames, stride, total_length):
                indices = [start + i * stride for i in range(num_frames)]
                # 确保索引不超过总长度
                if indices[-1] >= total_length:
                    # 如果最后一个索引超出范围，调整起始点
                    start = total_length - (num_frames * stride)
                    start = max(start, 0)
                    indices = [start + i * stride for i in range(num_frames)]
                return indices

            # 获取输入部分的帧索引
            batch_index_output = get_indices(start=0, num_frames=self.video_sample_n_frames, stride=stride, total_length=length_of_video)
        else:
            # 当视频长度小于样本帧数时，均匀重复帧
            interval = length_of_video / self.video_sample_n_frames
            batch_index_output = [min(int(i * interval), length_of_video - 1) for i in range(self.video_sample_n_frames)]

        batch_index_input = [batch_index_output[0]] * self.video_sample_n_frames

        return batch_index_input, batch_index_output

    def get_batch(self, data_info):
        data_type = data_info['type']

        video_id, pose_file = data_info['video_file_path'], data_info['camera_file_path']

        if self.data_root is None:
            video_dir = video_id
            pose_file_dir = pose_file
        else:
            video_dir = os.path.join(self.data_root, video_id)
            pose_file_dir = os.path.join(self.data_root, pose_file)

        whole_video, ori_h, ori_w = get_video_from_dir(video_dir)
        whole_camera_para = get_camera_from_dir(pose_file_dir)

        # print(len(whole_video))
        # print(len(whole_camera_para))

        assert len(whole_video) == len(whole_camera_para), "the length of video must be euqal with that of camera parameter."

        batch_index_input, batch_index_output = self.get_batch_index(len(whole_video))
        plucker_embedding_input, plucker_embedding_output = get_plucker_embedding(
            whole_camera_para,
            batch_index_input,
            batch_index_output,
            self.video_sample_size,
            ori_h,
            ori_w,
        )
        pixel_values_input, pixel_values_output = get_pixel_value(whole_video, batch_index_input, batch_index_output, self.video_transforms)

        return pixel_values_input, pixel_values_output, plucker_embedding_input, plucker_embedding_output, data_type

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.realestate_dataset_list[idx]
        sample = {}

        pixel_values_input, pixel_values_output, plucker_embedding_input, plucker_embedding_output, data_type = self.get_batch(data_info)
        sample["pixel_values_input"] = pixel_values_input
        sample["pixel_values_output"] = pixel_values_output
        sample["plucker_embedding_input"] = plucker_embedding_input
        sample["plucker_embedding_output"] = plucker_embedding_output
        sample['text'] = data_info['text']
        sample["data_type"] = data_type
        sample["idx"] = idx

        if self.enable_inpaint:
            mask = get_random_mask(pixel_values_output.size())
            mask_pixel_values = pixel_values_output * (1 - mask) + torch.ones_like(pixel_values_output) * -1 * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values_input"].permute(0, 2, 3, 1).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

            # ref_pixel_values = sample["pixel_values_input"][0].unsqueeze(0)
            # if (mask == 1).all():
            #     ref_pixel_values = torch.ones_like(ref_pixel_values) * -1
            # sample["ref_pixel_values"] = ref_pixel_values

        return sample


class VidGen(Dataset):
    def __init__(
        self,
        VidGen_dataset_list,
        data_root,
        video_sample_stride,
        video_sample_n_frames,
        video_sample_size,
        enable_inpaint=False,
    ):
        self.VidGen_dataset_list = VidGen_dataset_list
        self.data_root = data_root
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = video_sample_size
        self.enable_inpaint = enable_inpaint
        self.length = len(self.VidGen_dataset_list)

        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def get_batch_index(self, length_of_video):
        if length_of_video >= self.video_sample_n_frames:

            def compute_stride(num_frames, desired_stride, num_samples):
                max_possible_stride = num_frames // num_samples
                if max_possible_stride == 0:
                    return 1  # 最小步长为1
                return min(desired_stride, max_possible_stride)

            # 计算输入和输出部分的步长
            stride = compute_stride(length_of_video, self.video_sample_stride, self.video_sample_n_frames)

            def get_indices(start, num_frames, stride, total_length):
                indices = [start + i * stride for i in range(num_frames)]
                # 确保索引不超过总长度
                if indices[-1] >= total_length:
                    # 如果最后一个索引超出范围，调整起始点
                    start = total_length - (num_frames * stride)
                    start = max(start, 0)
                    indices = [start + i * stride for i in range(num_frames)]
                return indices

            # 获取输入部分的帧索引
            batch_index_output = get_indices(start=0, num_frames=self.video_sample_n_frames, stride=stride, total_length=length_of_video)
        else:
            # 当视频长度小于样本帧数时，均匀重复帧
            interval = length_of_video / self.video_sample_n_frames
            batch_index_output = [min(int(i * interval), length_of_video - 1) for i in range(self.video_sample_n_frames)]

        batch_index_input = [batch_index_output[0]] * self.video_sample_n_frames

        return batch_index_input, batch_index_output

    def get_batch(self, data_info):
        data_type = data_info['type']

        video_id = data_info['video_file_path']

        if self.data_root is None:
            video_dir = video_id
            # pose_file_dir = pose_file
        else:
            video_dir = os.path.join(self.data_root, video_id)
            # pose_file_dir = os.path.join(self.data_root, pose_file)

        whole_video, _, _ = get_video_from_dir(video_dir)
        # whole_camera_para = None

        # print(len(whole_video))
        # print(len(whole_camera_para))

        batch_index_input, batch_index_output = self.get_batch_index(len(whole_video))
        pixel_values_input, pixel_values_output = get_pixel_value(whole_video, batch_index_input, batch_index_output, self.video_transforms)
        plucker_embedding_input, plucker_embedding_output = get_empty_plucker_embedding(pixel_values_input, direction_number=6)

        return pixel_values_input, pixel_values_output, plucker_embedding_input, plucker_embedding_output, data_type

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.VidGen_dataset_list[idx]
        sample = {}

        pixel_values_input, pixel_values_output, plucker_embedding_input, plucker_embedding_output, data_type = self.get_batch(data_info)
        sample["pixel_values_input"] = pixel_values_input
        sample["pixel_values_output"] = pixel_values_output
        sample["plucker_embedding_input"] = plucker_embedding_input
        sample["plucker_embedding_output"] = plucker_embedding_output
        sample['text'] = data_info['text']
        sample["data_type"] = data_type
        sample["idx"] = idx

        if self.enable_inpaint:
            mask = get_random_mask(pixel_values_output.size())
            mask_pixel_values = pixel_values_output * (1 - mask) + torch.ones_like(pixel_values_output) * -1 * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values_input"].permute(0, 2, 3, 1).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

            # ref_pixel_values = sample["pixel_values_input"][0].unsqueeze(0)
            # if (mask == 1).all():
            #     ref_pixel_values = torch.ones_like(ref_pixel_values) * -1
            # sample["ref_pixel_values"] = ref_pixel_values

        return sample


class ALLDatasets(Dataset):

    DATASET_CLASSES = {
        'GenXD': GenXD,
        'kubric': KubricDataset,
        'objaverse': ObjaverseDataset,
        'realestate': RealEstateDataset,
        'VidGen': VidGen,
    }

    def __init__(
        self,
        ann_path,
        data_root=None,
        video_sample_size=[384, 672],
        video_sample_stride=4,
        video_sample_n_frames=16,
        enable_inpaint=False,
    ):
        print(f"loading annotations from {ann_path} ...")
        dataset = self._load_annotations(ann_path)
        self.data_root = data_root

        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.enable_inpaint = enable_inpaint

        # 按类型分组数据
        self.dataset_lists = self._split_datasets_by_type(dataset)

        # 初始化各子数据集
        self.sub_datasets = {}
        total_length = 0
        for dtype, dlist in self.dataset_lists.items():
            dataset_class = self.DATASET_CLASSES.get(dtype)
            if not dataset_class:
                print(f"Unsupported dataset type: {dtype}. Skipping.")
                continue
            self.sub_datasets[dtype] = dataset_class(
                dlist,
                self.data_root,
                self.video_sample_stride,
                self.video_sample_n_frames,
                self.video_sample_size,
                self.enable_inpaint,
            )
            total_length += len(self.sub_datasets[dtype])
            print(f"Loaded {len(self.sub_datasets[dtype])} samples for type '{dtype}'.")

        self.length = total_length
        print(f"Total data scale: {self.length}")

    def _load_annotations(self, ann_path: str) -> List[Dict[str, Any]]:
        """加载注释文件，支持CSV和JSON格式"""
        try:
            if ann_path.endswith('.csv'):
                with open(ann_path, 'r', encoding='utf-8') as csvfile:
                    return list(csv.DictReader(csvfile))
            elif ann_path.endswith('.json'):
                with open(ann_path, 'r', encoding='utf-8') as jsonfile:
                    return json.load(jsonfile)
            else:
                raise ValueError(f"Unsupported annotation file format: {ann_path}")
        except Exception as e:
            print(f"Failed to load annotations from {ann_path}: {e}")
            raise

    def _split_datasets_by_type(self, dataset: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """按类型分组数据"""
        dataset_lists: Dict[str, List[Dict[str, Any]]] = {dtype: [] for dtype in self.DATASET_CLASSES.keys()}
        for data in dataset:
            dtype = data.get('type')
            if dtype in dataset_lists:
                dataset_lists[dtype].append(data)
            else:
                print(f"Unknown dataset type '{dtype}' found in data. Skipping.")
        return dataset_lists

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.length}")

        # 遍历各子数据集，找到对应的样本
        cumulative_length = 0
        for dtype, dataset in self.sub_datasets.items():
            if idx < cumulative_length + len(dataset):
                sample = dataset[idx - cumulative_length]
                sample['source_type'] = dtype  # 添加来源类型信息
                sample['idx'] = idx
                sample['text'] = sample.get('text', '')  # 确保'text'字段存在
                return sample
            cumulative_length += len(dataset)

        # 如果未找到，抛出异常
        raise IndexError(f"Index {idx} not found in any sub-dataset.")


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_dataset = ALLDatasets(
        "datasets/z_mini_datasets/mini_datasets_metadata.json",
        'datasets/z_mini_datasets',
        video_sample_size=[384, 672],
        video_sample_stride=3,
        video_sample_n_frames=49,
        enable_inpaint=True,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    sample = train_dataset[325]
    print(sample)

    # # 检查所有样本是否有 None
    # for idx in range(len(train_dataset)):
    #     sample = train_dataset[idx]
    #     if sample is None:
    #         print(f"Dataset 返回了 None 在索引 {idx}")
