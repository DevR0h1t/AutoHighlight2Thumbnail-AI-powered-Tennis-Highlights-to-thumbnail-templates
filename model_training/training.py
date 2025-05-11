# model_training/training.py
# ------------------------
# Training script for:
#  1) YOLOv8 object detectors (players, ball)
#   2) ResNet-50 court-keypoint regressor

import argparse
import os
import torch
from ultralytics import YOLO
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Assume these modules exist in your scaffold:
from preprocessing.keypoints_dataset import KeypointsDataset
from models.court_keypoint_model import CourtKeypointModel


def train_detector(data_yaml, base_model, epochs, imgsz, project, name):
    """
    Train a YOLOv8 detector on the given dataset.

    Args:
        data_yaml (str): Path to YOLO data config (classes + train/val paths).
        base_model (str): Pretrained .pt to start from (e.g. 'yolov8n.pt').
        epochs (int): Number of training epochs.
        imgsz (int): Image size for training.
        project (str): Output folder for weights.
        name (str): Experiment name (subfolder under project).
    """
    model = YOLO(base_model)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=project,
        name=name,
        save=True
    )


def train_keypoints(data_dir, epochs, batch_size, lr, out_path):
    """
    Train a ResNet-50 head for court keypoint detection.

    Args:
        data_dir (str): Directory with images/labels for keypoints.
        epochs (int): Training epochs.
        batch_size (int): DataLoader batch size.
        lr (float): Learning rate.
        out_path (str): Where to save final .pt model.
    """
    # Dataset & DataLoader
    dataset = KeypointsDataset(root=data_dir)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = CourtKeypointModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for imgs, keypoints in loader:
            imgs = imgs.to(device)
            keypoints = keypoints.to(device)

            preds = model(imgs)
            loss = criterion(preds, keypoints)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        avg_loss = running_loss / len(dataset)
        print(f"Epoch {epoch}/{epochs}  Loss: {avg_loss:.4f}")

    # Save trained weights
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved keypoint model to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train detection or keypoint models.")
    parser.add_argument('--task', choices=['detector', 'keypoints', 'all'], default='all')
    parser.add_argument('--data_yaml', type=str, default='data/tennis_ball.yaml')
    parser.add_argument('--player_yaml', type=str, default='data/player.yaml')
    parser.add_argument('--keypoint_dir', type=str, default='data/court_keypoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--project', type=str, default='weights')
    parser.add_argument('--out_detector_name', type=str, default='yolov8n-tennis')
    parser.add_argument('--out_keypoints', type=str, default='weights/court_keypoints.pt')
    args = parser.parse_args()

    if args.task in ('detector', 'all'):
        # Train ball + player detector as a multi-class run
        train_detector(
            data_yaml=args.data_yaml,
            base_model='yolov8n.pt',
            epochs=args.epochs,
            imgsz=args.imgsz,
            project=args.project,
            name=args.out_detector_name
        )

    if args.task in ('keypoints', 'all'):
        train_keypoints(
            data_dir=args.keypoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            out_path=args.out_keypoints
        )
