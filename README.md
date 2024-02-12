# Video Prediction and Mask Segmentation

## Overview
This project focuses on the challenging tasks of video prediction and mask segmentation. It employs ConvLSTM (Convolutional Long Short-Term Memory) networks for predicting future frames in a video sequence and segments objects by generating masks. This approach can be particularly useful in applications such as video surveillance, autonomous driving, and dynamic scene understanding.

## Features
- **Video Prediction**: Uses ConvLSTM to predict future frames based on past sequences.
- **Mask Segmentation**: Segments objects in video frames to understand scene dynamics better.
- **Customizable Configurations**: Offers configuration options for prediction and segmentation tasks.
- **Dataset Sorting in Prediction**: Implements sorting of video folders in `PredictionDataset` for streamlined data processing.

## Project Structure
- `configs/`: Configuration files for prediction and segmentation models.
- `predictor/`: Implementation of the ConvLSTM predictor model.
- `segmenter/`: Implementation of the segmentation model.
- `utils/`: Utility scripts for dataset handling and other common functions.
- `predict_hidden.py`: Script for running predictions with hidden configurations.
- `requirements.txt`: Lists all the dependencies required to run the project.
- `train_predictor.py`: Script for training the ConvLSTM predictor model.
- `train_segmenter.py`: Script for training the segmentation model.

## Getting Started

### Prerequisites
Ensure you have Python 3.x installed on your system. You can install all the dependencies using:

```bash
pip install -r requirements.txt
