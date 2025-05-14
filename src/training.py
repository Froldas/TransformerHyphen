import logging
import numpy as np
import os
import shutil
import torch.nn as nn

from pathlib import Path
from torch import no_grad
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from src.utils import plot_loss


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                logging.info(f"Validation loss decreased ({self.best_loss:.6f} → {val_loss:.6f}).  Saving model ...")
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logging.info(f"No improvement in validation loss for {self.counter} epoch(s).")
            if self.counter >= self.patience:
                if self.verbose:
                    logging.info("Early stopping triggered.")
                self.early_stop = True

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


def model_training(model, train_dataset, num_epochs, num_folds, optimizer, loss_func, batch_size, device, work_dir, seed=42, early_stopping=False):
    train_losses = []
    val_losses = []

    early_stopping = EarlyStopping(patience=3, min_delta=0.001, verbose=False)
    checkpoints_dir = Path(work_dir) / "checkpoints"

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
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
            val_loss = validate_epoch(model, loss_func, fold_val_loader, device)
            val_losses.append(val_loss)

            if early_stopping:
                # Directory to save checkpoints
                checkpoint_dir = checkpoints_dir / f"fold_{fold + 1}"
                os.makedirs(checkpoint_dir, exist_ok=True)

                # Check early stopping condition
                early_stopping(val_loss, model)

                if early_stopping.early_stop:
                    logging.info("Early stopping activated. Restoring best model weights.")
                    model.load_state_dict(early_stopping.best_model_state)
                    shutil.rmtree(checkpoints_dir)
                    plot_loss(train_losses, save_path=Path(work_dir) / "train_curve.png")
                    plot_loss(val_losses, title="Validation Loss", save_path=Path(work_dir) / "val_curve.png")
                    return

    plot_loss(train_losses, save_path=Path(work_dir) / "train_curve.png")
    plot_loss(val_losses, title="Validation Loss", save_path=Path(work_dir) / "val_curve.png")


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

