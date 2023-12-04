#!/usr/bin/env python3

import os
from time import time
from datetime import timedelta
import imageio.v3 as iio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SegmentationDataset(Dataset):
    def __init__(self, datadir, transform=None):
        """
        Creates a pytorch dataset from the dataset folders

        Args:
            datadir (str): Path to root folder of dataset
            transform : transformations
        """
        self.datadir = datadir
        self.transform = transform
        self.data = []

        vid_dirs = [f.path for f in os.scandir(self.datadir) if f.is_dir()]

        for vid_path in vid_dirs:
            image_paths = [
                os.path.join(vid_path, f) for f in os.listdir(vid_path)
                if f.endswith('.png')
            ]

            mask_path = os.path.join(vid_path, "mask.npy")
            mask = np.load(mask_path)

            for image_path in image_paths:
                idx = int(image_path.split("_")[-1].strip(".png"))
                img = iio.imread(image_path)
                img_tensor = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1)  # output shape = n_channels x H x W # noqa
                mask_i = mask[idx]
                mask_tensor = torch.from_numpy(mask_i)

                self.data.append([img_tensor, mask_tensor])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, mask = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, mask


class PredictionDataset(Dataset):
    def __init__(self, datadir, transform=None):
        """
        Creates a pytorch dataset from all the dataset folders

        Args:
            datadir (str): Path to root folder of dataset
        """
        self.datadir = datadir
        self.transform = transform
        self.data = []

        vid_dirs = [f.path for f in os.scandir(self.datadir) if f.is_dir()]

        for vid_path in vid_dirs:
            image_paths = [
                f for f in os.listdir(vid_path) if f.endswith('.png')
            ]
            sorted_image_paths = sorted(
                image_paths,
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            vid_arr = []

            for img_path in sorted_image_paths:
                img_path = os.path.join(vid_path, img_path)
                img = iio.imread(img_path)
                vid_arr.append(img)

            vid_np_array = np.array(vid_arr)
            vid_tensor = torch.from_numpy(vid_np_array)

            self.data.append(vid_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_tensor = self.data[idx]
        if self.transform:
            video_tensor = self.transform(video_tensor)
        return video_tensor


def _create_dataloader(
    dataset_dir, dataset_cls, batch_size, transform=None, shuffle=True
):
    """Creates torch dataloader

    Args:
        dir (str): absolute path to data samples
        dataset_cls (torch.utils.data.Dataset): dataset class to use
        batch_size (int): Batch size
        transform:
        shuffle (bool): shuffle data samples if True
    Returns:
        torch.utils.data.DataLoader
    """
    t1 = time()
    dataset = dataset_cls(dataset_dir, transform=None)
    print("Started creating train_dataloader ...")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    elapsed = str(timedelta(seconds=time() - t1))
    print(
        f"Created {dataset_cls.__name__} dataloader",
        f"from {dataset_dir} in {elapsed}s"
    )

    return dataloader
