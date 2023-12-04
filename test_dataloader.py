#! /usr/bin/env python3

import os
import torch
import argparse
from utils.dataloaders import SegmentationDataset, PredictionDataset
from utils.dataloaders import _create_dataloader


def get_commandline_args():
    """Get commandline arguments

    Returns:
        argparse.Namespace: a dict-type object to access arguments
    """
    def is_valid_path(parser, arg):
        """Checks if the passed argument is a valid file / directory"""
        if not os.path.exists(arg):
            parser.ArgumentTypeError(
                "The passed directory / file %s does not exist!" % arg
            )
        else:
            return os.path.abspath(arg)     # return absolute path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        help="dataset directory, contains (train, val, unlabeled dirs)",
        type=lambda x: is_valid_path(parser, x)
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Getting the data
    args = get_commandline_args()
    dataset_dir = args.dataset
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    transform = None
    batch_size = 10
    seg_train_dataloader = _create_dataloader(
        train_dir, SegmentationDataset, batch_size, transform, shuffle=True
    )
    pred_train_dataloader = _create_dataloader(
        train_dir, PredictionDataset, batch_size, transform, shuffle=True
    )

    # Example
    first_seg_batch = next(iter(seg_train_dataloader))
    images, masks = first_seg_batch

    video = next(iter(pred_train_dataloader))

    try:
        assert images.shape == torch.Size((batch_size, 160, 240, 3))
    except AssertionError:
        raise Exception(
            "Expected images.shape %r to have size [%d, 160, 240, 3]"
            % (images.shape, batch_size)
        )
    finally:
        print(
            "images.shape (%r) match expectation [%d, 160, 240, 3]"
            % (images.shape, batch_size)
        )
    try:
        assert masks.shape == torch.Size((batch_size, 160, 240))
    except AssertionError:
        raise Exception(
            "Expected masks.shape %r to have size [%d, 160, 240]"
            % (masks.shape, batch_size)
        )
    finally:
        print(
            "masks.shape (%r) match expectation [%d, 160, 240]"
            % (masks.shape, batch_size)
        )
    try:
        assert video.shape == torch.Size((batch_size, 22, 160, 240, 3))
    except AssertionError:
        raise Exception(
            "Expected video.shape %r to have size [%d, 22, 160, 240, 3]"
            % (video.shape, batch_size)
        )
    finally:
        print(
            "video.shape (%r) match expectation [%d, 22, 160, 240, 3]"
            % (images.shape, batch_size)
        )
