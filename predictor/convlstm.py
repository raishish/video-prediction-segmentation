#! /usr/bin/env python3

from time import time
from datetime import timedelta
import torch
import torch.nn as nn
import torch.nn.init as init


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Original ConvLSTM cell as proposed by Shi et al.
class ConvLSTMCell(nn.Module):
    def __init__(
        self, in_channels, out_channels,
        kernel_size, padding,
        activation, frame_size
    ):
        """
        Args:
            in_channels (int): input channels for conv3d
            out_channels (int): output channels for conv3d
            kernel_size (tuple): kernel size for conv3d
            padding (tuple): padding for conv3d
            activation (str): one of ["tanh", "relu"]
            frame_size (tuple): dimensions (H, W) of input images
        """
        super(ConvLSTMCell, self).__init__()

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding
        )

        # Initialize convolutional layer using Xavier initialization
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

        # Initialize custom parameters using initialization strategy
        init.xavier_uniform_(self.W_ci)
        init.xavier_uniform_(self.W_co)
        init.xavier_uniform_(self.W_cf)

    def forward(self, X, H_prev, C_prev):
        """
        Args:
            X (torch.Tensor): current input to ConvLSTMCell
            H_prev (torch.Tensor): Previous hidden representation
            C_prev (torch.Tensor): Previous output
        Returns:
            (H, C): Current hidden representation, current output
        """
        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, C_conv, o_conv = torch.chunk(
            conv_output, chunks=4, dim=1
        )

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev)

        # Current Cell output
        C = forget_gate * C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C)

        # Current Hidden State
        H = output_gate * self.activation(C)
        return H, C


class ConvLSTM(nn.Module):
    def __init__(
        self, in_channels, out_channels,
        kernel_size, padding,
        activation, frame_size
    ):
        """
        Args:
            in_channels (int): input channels for conv3d
            out_channels (int): output channels for conv3d
            kernel_size (tuple): kernel size for conv3d
            padding (tuple): padding for conv3d
            activation (str): one of ["tanh", "relu"]
            frame_size (tuple): dimensions (H, W) of input images
        """
        super(ConvLSTM, self).__init__()
        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(
            in_channels, out_channels,
            kernel_size, padding,
            activation, frame_size
        )

        # Initialize ConvLSTMCell using Xavier initialization
        for param in self.convLSTMcell.parameters():
            if len(param.shape) > 1:  # Check if the parameter is a weight
                init.xavier_uniform_(param)

    def forward(self, X):
        """
        Args:
            X: frame sequence (batch_size, num_channels, seqlen, height, width)
        Returns:
            torch.Tensor: Current output
        """
        batch_size, _, seqlen, height, width = X.size()

        # Initialize output
        output = torch.zeros(
            batch_size, self.out_channels, seqlen, height, width, device=device
        )

        # Initialize Hidden State
        H = torch.zeros(
            batch_size, self.out_channels, height, width, device=device
        )

        # Initialize Cell Input
        C = torch.zeros(
            batch_size, self.out_channels, height, width, device=device
        )

        # Unroll over time steps
        for time_step in range(seqlen):
            H, C = self.convLSTMcell(X[:, :, time_step], H, C)
            output[:, :, time_step] = H

        return output


class Seq2Seq(nn.Module):
    def __init__(
        self, num_channels, num_kernels,
        kernel_size, padding,
        activation, frame_size, num_layers
    ):
        """
        Sequence to Sequence Model

        Args:
            in_channels (int): input channels for conv3d
            out_channels (int): output channels for conv3d
            kernel_size (tuple): kernel size for conv3d
            padding (tuple): padding for conv3d
            activation (str): one of ["tanh", "relu"]
            frame_size (tuple): dimensions (H, W) of input images
        """
        super(Seq2Seq, self).__init__()
        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding,
                activation=activation, frame_size=frame_size)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        )

        # Add rest of the layers
        for layer in range(2, num_layers + 1):
            self.sequential.add_module(
                f"convlstm{layer}", ConvLSTM(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding,
                    activation=activation, frame_size=frame_size
                )
            )
            self.sequential.add_module(
                f"batchnorm{layer}", nn.BatchNorm3d(num_features=num_kernels)
            )

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding
        )

    def forward(self, X):
        """
        Args:
            X (torch.Tensor): input seed frame sequence
        """
        output = self.sequential(X)
        # Return only the last output frame
        output = self.conv(output[:, :, -1])
        output = nn.Sigmoid()(output) * 255    # normalise and scale output

        return output


def _process_one_batch(
    data: torch.Tensor,
    batch_num: int,
    model: torch.nn.Module,
    device: torch.device,
    mode: str,
    optimizer,
    loss_criterion,
    acc_criterion=None,
    verbose: int = 0
):
    """
    Process one batch of training or validation data

    # NOTE: Only supports batch_size = 1

    Args:
        data (torch.Tensor): One batch size worth data
                             shape = [batch_size, 22, 160, 240, 3]
        batch_num (int): batch iteration
        model (torch.nn.Module): model to train / evaluate
        device (torch.device): torch device
        mode (str): one of ["train", "eval"]
        optimizer (torch.optim): Optimizer
        loss_criterion (nn.modules.loss): Loss criterion
        acc_criterion (): Accuracy criterion
        verbose (int): verbosity of logs
    Returns:
        loss, accuracy (optional)
    """
    t1 = time()
    batch_loss = 0
    batch_acc = 0
    data = data.to(device)

    # Get the first (only) video in the batch (batch_size = 1)
    pred_video = torch.zeros_like(data)
    pred_video = pred_video.to(device)
    pred_video[:, :11, :] = data[:, :11, :]  # first input = first 11 frames

    for timestep in range(11):
        input_frames = pred_video[:, timestep: timestep + 11, :].permute(0, 4, 1, 2, 3)
        # target_frame = 12th frame from beginning of input
        target_frame = data[:, 11 + timestep, :].float()

        pred_frame = model(input_frames)
        pred_frame = pred_frame.permute(0, 2, 3, 1)
        pred_video[0, timestep + 11, :] = pred_frame

        if mode == "train":
            # calculate loss per predicted frame and backprop
            loss = torch.sqrt(
                loss_criterion(pred_frame.flatten(), target_frame.flatten())
            )
            batch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Calculate (val) loss and accuracy on last predicted frame
    if mode == "eval":
        batch_loss = torch.sqrt(
            loss_criterion(pred_frame.flatten(), target_frame.flatten())
        )
    if acc_criterion:
        batch_acc = acc_criterion(pred_frame, target_frame)

    elapsed = str(timedelta(seconds=time() - t1))
    if verbose > 1:
        print(
            f"[{mode}] Batch {batch_num} ",
            f"| Loss = {batch_loss.item()} ",
            f"| Time = {elapsed}s"
        )

    return batch_loss, batch_acc
