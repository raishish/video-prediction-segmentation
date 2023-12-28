# !/usr/bin/env python3

import torch
import os
import json
import numpy as np
import argparse
from predictor.convlstm import Seq2Seq
from segmenter.segmenter import VisionTransformer, MaskTransformer, Segmenter
from utils.dataloaders import PredictionDataset, _create_dataloader


def get_models(
    predictor_params: dict,
    segmenter_params: dict,
    predictor_checkpoint_path: str,
    segmenter_checkpoint_path: str,
    device: torch.device
):
    """
    Args:
        predictor_params: Predictor parameters
        segmenter_params: Segmenter parameters
        predictor_checkpoint_path: (.pth) path to load predictor model from
        segmenter_checkpoint_path: (.pth) path to load segmenter model from
        device: device to load model on
    Returns:
        [torch.nn.Module, torch.nn.Module]: Predictor, Segmenter models
    """
    predictor = Seq2Seq(
        num_channels=predictor_params["num_channels"],
        num_kernels=predictor_params["num_kernels"],
        kernel_size=predictor_params["kernel_size"],
        padding=predictor_params["padding"],
        activation=predictor_params["activation"],
        frame_size=predictor_params["frame_size"],
        num_layers=predictor_params["num_layers"]
    )
    pred_chkpt = torch.load(predictor_checkpoint_path, map_location=device)
    if isinstance(pred_chkpt, dict) and 'model' in pred_chkpt:
        predictor.load_state_dict(pred_chkpt['model'])
    else:   # if it is a weight file
        predictor.load_state_dict(pred_chkpt)
    print("Loaded predictor from checkpoint")

    seg_encoder = VisionTransformer(
        image_size=segmenter_params["im_size"],
        patch_size=segmenter_params["patch_size"],
        n_layers=segmenter_params["encoder_num_layers"],
        d_model=segmenter_params["encoder_embed_dim"],
        d_ff=segmenter_params["encoder_mlp_dim"],
        n_heads=segmenter_params["encoder_num_heads"],
        n_cls=segmenter_params["num_classes"],
        dropout=segmenter_params["encoder_dropout"],
        channels=segmenter_params["in_channels"]
    )
    seg_decoder = MaskTransformer(
        n_cls=segmenter_params["num_classes"],
        patch_size=segmenter_params["patch_size"],
        d_encoder=segmenter_params["encoder_embed_dim"],
        n_layers=segmenter_params["decoder_num_layers"],
        n_heads=segmenter_params["decoder_num_heads"],
        d_model=segmenter_params["decoder_embed_dim"],
        d_ff=segmenter_params["decoder_mlp_dim"],
        dropout=segmenter_params["decoder_dropout"]
    )
    segmenter = Segmenter(
        seg_encoder,
        seg_decoder,
        n_cls=segmenter_params["num_classes"]
    )

    seg_chkpt = torch.load(segmenter_checkpoint_path, map_location=device)
    if isinstance(seg_chkpt, dict) and 'model' in seg_chkpt:
        segmenter.load_state_dict(seg_chkpt['model'])
    else:   # if it is a weight file
        segmenter.load_state_dict(seg_chkpt)
    print("Loaded segmenter from checkpoint")

    return predictor, segmenter


# Compute Mean IOU
def _process_one_batch(vid_tensor, model_p, model_s, device):
    """
    Process one batch of training or validation data

    Args:
        vid_tensor (torch.Tensor): One batch size worth data
                             shape = [batch_size, 11, 160, 240, 3]
        target_tensor (torch.Tensor): [160, 240, 3]
        mask_tensor (torch.Tensor): [160, 240]

        model (torch.nn.Module): model to train / evaluate
        device (torch.device): torch device
        verbose (int): verbosity of logs
        loss_criterion (nn.modules.loss): Loss criterion
        acc_criterion (): Accuracy criterion
    Returns:
        loss, accuracy (optional)
    """
    pred_video = torch.zeros(1, 22, 160, 240, 3)
    pred_video = pred_video.to(device)
    pred_video[0, :11, :, :, :] = vid_tensor
    pred = None

    with torch.no_grad():
        for timestep in range(11):
            # Add previously predicted frame to the input

            # Preparing input and target tensors

            input = pred_video.squeeze()[0+timestep:11+timestep, :, :, :].permute(3, 0, 1, 2)
            input = input.unsqueeze(0).to(device)

            # Forward pass
            output = model_p(input).float()
            pred = output.squeeze(0).permute(1, 2, 0)
            pred_video[0, 11 + timestep, :, :, :] = pred

        input_frame = pred.unsqueeze(0).permute(0, 3, 1, 2)
        input_frame = input_frame.to(device)
        pred_mask = model_s(input_frame).argmax(dim=1)
        pred_mask = pred_mask.squeeze()

    return pred_mask


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
        "--datadir",
        "-d",
        help="hidden dataset directory, contains video folders with image frames",
        type=lambda x: is_valid_path(parser, x),
        required=True
    )
    parser.add_argument(
        "--save-dir",
        "-s",
        help="Directory to save models / checkpoints",
        type=lambda x: is_valid_path(parser, x),
        required=True
    )
    parser.add_argument(
        "--pred-chkpt",
        "-p",
        help="path to predicted model checkpoint to inference from",
        type=lambda x: is_valid_path(parser, x),
        required=True
    )
    parser.add_argument(
        "--seg-chkpt",
        "-c",
        help="path to segmenter model checkpoint to inference from",
        type=lambda x: is_valid_path(parser, x),
        required=True
    )
    parser.add_argument(
        "--model-specs",
        "-m",
        help="path to json file containing model specifications",
        type=lambda x: is_valid_path(parser, x),
        required=True
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity (-v, -vv, etc)",
        required=False
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_commandline_args()
    hidden_dataset_path = args.datadir
    save_file_folder = args.save_dir
    pred_chkpt_path = args.pred_chkpt
    seg_chkpt_path = args.seg_chkpt
    model_specs_path = args.model_specs
    batch_size = 1

    verbose = args.verbose

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(model_specs_path) as fd:
        model_specs_dict = json.load(fd)
    predictor_params = model_specs_dict["models"]["predictor"]["ConvLSTM"]
    segmenter_params = model_specs_dict["models"]["segmenter"]["Segmenter"]

    print("Loading Models...")
    predictor, segmenter = get_models(
        predictor_params=predictor_params,
        segmenter_params=segmenter_params,
        predictor_checkpoint_path=pred_chkpt_path,
        segmenter_checkpoint_path=seg_chkpt_path,
        device=device
    )

    predictor.to(device)
    predictor.eval()

    segmenter.to(device)
    segmenter.eval()

    print("Loading Hidden Data...")
    transform = None
    hidden_dataloader = _create_dataloader(
        hidden_dataset_path, PredictionDataset, batch_size, transform, shuffle=False
    )

    print("Start Inferencing")
    final_list = []

    for batch, data in enumerate(hidden_dataloader):
        vid_tensor = data
        pred_mask = _process_one_batch(vid_tensor, predictor, segmenter, device)
        pred_np = pred_mask.cpu().numpy()
        final_list.append(pred_np)

    final_np = np.array(final_list)
    file_path = os.path.join(save_file_folder, "predicted_mask.npy")
    print("Saving Final Tensor at: ", file_path)
    np.save(file_path, final_np)

    print("Final Numpy Array shape: ", final_np.shape)
    print("Final Numpy dtype: ", final_np.dtype)
