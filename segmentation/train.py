import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from model import MiniUNet
from dataset import RGBDataset
from utils import iou, save_checkpoint, load_checkpoint, save_predictions, plot_learning_curve, check_dataset, check_dataloader


def run_epoch(model, device, dataloader, criterion, optimizer=None):
    """
    Run a single epoch of training or validation.
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    epoch_loss, epoch_iou, data_size = 0, 0, 0

    for batch in dataloader:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device).long()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        batch_size = inputs.size(0)

        epoch_loss += loss.item() * batch_size
        epoch_iou += np.sum(iou(outputs, targets))
        data_size += batch_size

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return epoch_loss / data_size, epoch_iou / data_size


def main():
    # Check for device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directories setup
    root_dir = Path('./dataset/')
    train_dir = root_dir / 'train'
    val_dir = root_dir / 'val'
    test_dir = root_dir / 'test'

    # Load datasets and dataloaders
    BATCH_SIZE = 4
    NUM_WORKERS = 2
    train_dataset = RGBDataset(train_dir, ground_truth=True)
    val_dataset = RGBDataset(val_dir, ground_truth=True)
    test_dataset = RGBDataset(test_dir, ground_truth=False)
    # check_dataset(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    # check_dataloader(train_loader)

    # Initialize model, loss, and optimizer
    model = MiniUNet().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training and validation loop
    num_epochs = 15
    best_miou = float('-inf')
    history = {'train_loss': [], 'train_miou': [], 'val_loss': [], 'val_miou': []}

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')

        train_loss, train_miou = run_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_miou = run_epoch(model, device, val_loader, criterion)

        history['train_loss'].append(train_loss)
        history['train_miou'].append(train_miou)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_miou)

        print(f"Train loss/mIoU: {train_loss:.2f}/{train_miou:.2f}")
        print(f"Validation loss/mIoU: {val_loss:.2f}/{val_miou:.2f}")

        if val_miou > best_miou:
            best_miou = val_miou
            save_checkpoint(model, epoch, val_miou)

    # Load the best model and make predictions
    model, best_epoch, best_miou = load_checkpoint(model)
    save_predictions(model, device, val_loader, val_dir)
    save_predictions(model, device, test_loader, test_dir)

    # Plot and save learning curves
    plot_learning_curve(history['train_loss'], history['train_miou'], history['val_loss'], history['val_miou'])


if __name__ == '__main__':
    main()
