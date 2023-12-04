#! /usr/bin/env python3

import os
import argparse
import torch
from torch import nn
from time import time
from datetime import timedelta
from torch.optim import Adam

from utils.dataloaders import PredictionDataset, _create_dataloader
from predictor.convlstm import Seq2Seq, _process_one_batch


def train(
    model: torch.nn.Module,
    device: torch.device,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    num_epochs: int,
    optimizer,
    loss_criterion,
    acc_criterion=None,
    patience: int = 10,
    save_dir: str = os.path.abspath("."),
    checkpoint_path: str = None,
    verbose: int = 0
):
    """
    Trains the model

    Args:
        model (torch.nn.module): pytorch model
        device (torch.device): Device for training
        train_dataloader (torch.Dataloader): Train dataloader
        val_dataloader (torch.Dataloader): Validation dataloader
        num_epochs (int): number of epochs to train
        optimizer (torch.optim): Optimizer
        loss_criterion (nn.modules.loss): Loss criterion
        acc_criterion: Accuracy criterion
        patience (int): Number of epochs to wait for val_acc to improve
                        before breaking training loop
        save_dir (str): location to save model if val_acc improves
        checkpoint_path (str): location to load saved model from, it if exists
        verbose (int): verbosity of logs
    """
    model.to(device)
    best_val_loss = 0
    counter = 0  # Counter for early stopping
    epoch_begin = 0
    num_train_batches = len(train_dataloader)
    num_val_batches = len(val_dataloader)

    if checkpoint_path and os.path.exists(os.path.abspath(checkpoint_path)):
        checkpoint_path = os.path.abspath(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_begin = checkpoint['epoch'] + 1   # epoch to start training with
        best_val_acc = checkpoint['best_val_acc']
        print(f"Model resumed from checkpoint at {checkpoint_path}")

    for epoch in range(epoch_begin, epoch_begin + num_epochs):
        # Train the model
        t1 = time()
        if verbose > 0:
            print("=" * 25, f"Epoch {epoch + 1}", "=" * 25, "\n")
        model.train()
        total_train_loss = 0
        total_train_acc = 0

        for batch_num, data in enumerate(train_dataloader, 1):
            batch_loss, batch_acc = _process_one_batch(
                data, batch_num, model, device, "train",
                optimizer, loss_criterion, acc_criterion=acc_criterion,
                verbose=verbose
            )
            total_train_loss += batch_loss
            total_train_acc += batch_acc

        elapsed = str(timedelta(seconds=time() - t1))
        if verbose > 0:
            print(f"\nProcessed training epoch {epoch + 1} in {elapsed}")
            print("-" * 60, "\n")

        avg_train_loss = total_train_loss / num_train_batches
        if acc_criterion:
            avg_train_acc = total_train_acc / num_train_batches

        # Validate the model
        t2 = time()
        model.eval()
        total_val_loss = 0
        total_val_acc = 0
        if verbose > 0:
            print("\n", "-" * 25, "Validation", "-" * 25, "\n")

        with torch.no_grad():
            for batch_num, data in enumerate(val_dataloader, 1):
                batch_loss, batch_acc = _process_one_batch(
                    data, batch_num, model, device, "eval",
                    optimizer, loss_criterion, acc_criterion=acc_criterion,
                    verbose=verbose
                )
            total_val_loss += batch_loss
            total_val_acc += batch_acc

        avg_val_loss = total_val_loss / num_val_batches
        if acc_criterion:
            avg_val_acc = total_val_acc / num_val_batches

        elapsed = str(timedelta(seconds=time() - t2))
        if verbose > 0:
            print(f"\nProcessed validation epoch {epoch + 1} in {elapsed}\n")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0  # Reset the counter when there's an improvement
            best_model_path = os.path.join(
                save_dir, "best_convlstm_model.pth"
            )
            checkpoint = {
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            best_checkpoint_path = os.path.join(
                save_dir, "best_convlstm_checkpoint.pth"
            )
            print(
                f"Found better validation accuracy ({best_val_acc}),"
                f"saving model to {best_model_path}",
                f"saving checkpoint to {best_checkpoint_path}"
            )

            torch.save(model.state_dict(), best_model_path)
            torch.save(checkpoint, best_checkpoint_path)
        else:
            counter += 1

        epoch_metrics = f"\nEpoch: {epoch + 1} - " + \
                        f"Training loss: {avg_train_loss:.4f} - " + \
                        (f"Training acc: {avg_train_acc * 100:.2f}% - " if acc_criterion else '') + \
                        f"Validation loss: {avg_val_loss:.4f} - " + \
                        (f"Validation acc: {avg_val_acc * 100:.2f}% - " if acc_criterion else '') + \
                        f"Best Validation Loss: {best_val_loss:.4f}"

        print(epoch_metrics)
        print("=" * 60, "\n")

        # Early stopping condition
        if counter >= patience:
            print(
                "No improvement in validation accuracy for {patience} epochs.",
                "Early stopping..."
            )
            break


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
        type=lambda x: is_valid_path(parser, x),
        required=True
    )
    parser.add_argument(
        "--save-dir",
        "-s",
        help="Directory to save models / checkpoints",
        type=lambda x: is_valid_path(parser, x),
        required=False
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        help="path to model checkpoint to resume training from",
        type=lambda x: is_valid_path(parser, x),
        required=False
    )
    parser.add_argument(
        "--batchsize",
        "-b",
        help="Batchsize for training",
        type=int,
        required=False
    )
    parser.add_argument(
        "--epochs",
        "-e",
        help="Number of epochs for training",
        type=int,
        required=False
    )
    parser.add_argument(
        "--output-file",
        "-o",
        help="stdout file",
        required=False
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
    dataset_dir = args.dataset
    save_dir = args.save_dir
    checkpoint_path = args.checkpoint
    verbose = args.verbose

    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    transform = None
    if args.batchsize:
        batch_size = args.batchsize
    else:
        batch_size = 8

    train_dataloader = _create_dataloader(
        train_dir, PredictionDataset, batch_size, transform, shuffle=True
    )
    val_dataloader = _create_dataloader(
        val_dir, PredictionDataset, batch_size, transform, shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model params
    im_size = (160, 240)
    in_channels = 3
    kernel_size = (3, 3)
    num_kernels = 64
    padding = (1, 1)
    activation = "relu"
    num_layers = 3

    # Initialize the model
    model = Seq2Seq(
        num_channels=in_channels,
        num_kernels=64,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation,
        frame_size=im_size,
        num_layers=num_layers
    )
    print("Model instantiated.")

    # Training params
    loss_criterion = nn.MSELoss()
    num_epochs = 20
    optimizer = Adam(model.parameters(), lr=1e-4)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    patience = 10  # Number of epochs to wait for improvement

    if args.epochs:
        num_epochs = args.epochs
    else:
        num_epochs = 200

    train(
        model, device,
        train_dataloader, val_dataloader,
        num_epochs, optimizer,
        loss_criterion, acc_criterion=None, patience=patience,
        save_dir=save_dir, checkpoint_path=checkpoint_path,
        verbose=verbose
    )
