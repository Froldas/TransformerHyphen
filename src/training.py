
import logging
import numpy as np
import torch.nn as nn

from pathlib import Path
from torch import no_grad
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from src.utils import plot_loss


def train_epoch(model: nn.Module, train_loader, optimizer, loss_func, device):
    model.train()
    epoch_loss = []

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Forward pass
        outputs = model(batch_X)
        loss = loss_func(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(float(loss))
    return epoch_loss


def model_training(model, train_dataset, num_epochs, num_folds, optimizer, loss_func, batch_size, device, work_dir):
    train_losses = []
    eval_losses = []
    kf = KFold(n_splits=num_folds)
    for epoch in range(num_epochs):
        for fold, (train_split_idx, val_split_idx) in enumerate(kf.split(range(len(train_dataset)))):
            fold_train_subset = Subset(train_dataset, train_split_idx)
            fold_val_subset = Subset(train_dataset, val_split_idx)

            fold_train_loader = DataLoader(fold_train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
            fold_val_loader = DataLoader(fold_val_subset, batch_size=batch_size)

            epoch_loss = train_epoch(model, fold_train_loader, optimizer, loss_func, device)
            train_losses.append(np.mean(epoch_loss))
            logging.info(
                f'Round [{num_folds * epoch + fold + 1}/{num_folds * num_epochs}], Loss: {np.mean(epoch_loss):.4f}')
            eval_losses.append(validate_epoch(model, loss_func, fold_val_loader, device))

    plot_loss(train_losses, save_path=Path(work_dir) / "train_curve.png")
    plot_loss(eval_losses, title="Validation Loss", save_path=Path(work_dir) / "validation_curve.png")


def validate_epoch(model: nn.Module, loss_func, validation_loader, device):
    model.eval()
    val_final_loss = 0.0
    with no_grad():
        val_loss = []
        for batch_X, batch_y in validation_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = loss_func(predictions, batch_y)
            val_loss.append(float(loss))
        val_final_loss = np.mean(val_loss)
        logging.info(f'Val loss: {val_final_loss:.4f}')

    return val_final_loss

