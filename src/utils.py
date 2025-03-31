import logging
import numpy as np
import os
import random
import torch.nn as nn
import torch
import yaml
import sys

from pathlib import Path
from torch import no_grad, manual_seed, save
from torch.utils.data import DataLoader, Subset
from torchview import draw_graph
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from src.constants import HYPHENS


def load_yaml_conf(path: str | os.PathLike):
    with open(path, encoding='utf8') as stream:
        try:
            return yaml.safe_load(stream, )
        except yaml.YAMLError as exc:
            print(exc)


def set_seed(seed: int):
    """
    Fix the seed for all libraries to guarantee reproducibility
    :param seed: integer seed value
    :return: None
    """
    manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def setup_logger(log_path):
    """
    Set up the logger and remove all logs
    :param log_path: filepath to the logging file
    :return: None
    """

    # remove all logging set up before

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # clean the previous log file if exists
    Path.unlink(log_path, missing_ok=True)

    # Configure the logger
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO
        format='%(asctime)s - %(message)s',  # Define the log message format
        datefmt='%Y-%m-%d %I:%M:%S',
        handlers=[
            logging.FileHandler(log_path, "w+", 'utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


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


def validate(model: nn.Module, loss_func, validation_loader, device):
    model.eval()
    with no_grad():
        val_loss = []
        for batch_X, batch_y in validation_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = loss_func(predictions, batch_y)
            val_loss.append(float(loss))
        logging.info(f'Val loss: {np.mean(val_loss):.4f}')


def remove_hyphenation(string):
    return string.translate({ord(hyph): None for hyph in HYPHENS})


def insert_hyphenation(string, bit_list):
    # Convert the string to a list for easier manipulation
    string_list = list(string)

    # Iterate over the bit_list and insert '-' where there is a 1
    hyphens_inserted = 0
    for index, bit in enumerate(bit_list):
        if bit == 1:
            # Insert '-' at the predicted spot
            string_list.insert(index + hyphens_inserted + 1, '-')
            hyphens_inserted += 1

    # Convert the list back to a string
    result_string = ''.join(string_list)
    return result_string


def save_model(model, path: Path):
    os.makedirs(path.parent, exist_ok=True)
    save(model.state_dict(), path)
    logging.info(f"Model saved to {path}")


def visualize(model, dataset, work_dir):
    model_graph = draw_graph(model, input_size=(1, dataset.input_size), expand_nested=True)
    model_graph.visual_graph.render(filename="model", format='pdf', directory=work_dir, quiet=True)


def dump_dataset(dataset, indices, output_path):
    Path.unlink(output_path, missing_ok=True)
    with open(output_path, "w+", encoding="utf-8") as f:
        for index in indices:
            f.writelines(f"{dataset.words[index]}\n")


def split_dataset(dataset, train_split, work_dir=None, dump_datasets=False):
    train_dataset_idx, test_dataset_idx = train_test_split(list(range(len(dataset))), test_size=(1.0 - train_split))

    train_dataset = Subset(dataset, train_dataset_idx)
    val_dataset = Subset(dataset, test_dataset_idx)

    if dump_datasets:
        dump_dataset(dataset, train_dataset_idx, Path(work_dir) / "train_dataset.wlh")
        dump_dataset(dataset, test_dataset_idx, Path(work_dir) / "test_dataset.wlh")

    return train_dataset, val_dataset


def model_size(model):
    torch.save(model.state_dict(), "temp.pth")
    size = os.path.getsize("temp.pth") / 1024  # Convert to KB
    os.remove("temp.pth")
    return size


def model_evaluation(model, X, y, dataset, label="Full model"):
    x_pred = model(torch.Tensor(np.array(X)).to("cpu"))

    accuracy = accuracy_score(torch.Tensor(np.array(y)).detach().numpy(),
                              x_pred.to("cpu").detach().numpy())
    recall = recall_score(torch.Tensor(np.array(y)).detach().numpy(), x_pred.to("cpu").detach().numpy(),
                          average="samples", zero_division=0.0)
    precision = precision_score(torch.Tensor(np.array(y)).detach().numpy(), x_pred.to("cpu").detach().numpy(),
                                average="samples", zero_division=0.0)
    f1 = f1_score(torch.Tensor(np.array(y)).detach().numpy(), x_pred.to("cpu").detach().numpy())

    dataset_size_kb = os.path.getsize(dataset) / 1024
    model_size_kb = model_size(model)
    logging.info(f"{label} evaluation: ")
    logging.info(f"    Dataset size is: {dataset_size_kb:.2f} KB")
    logging.info(f"    {label} size: {model_size_kb:.2f} KB")
    logging.info(f"    {label} Efficiency: {(dataset_size_kb / model_size_kb) * 100:.2f} %")
    logging.info(f"    {label} Accuracy: {accuracy:.4f}")
    logging.info(f"    {label} Recall: {recall:.4f}")
    logging.info(f"    {label} Precision: {precision:.4f}")
    logging.info(f"    {label} F1: {f1:.4f}")


def model_training(model, train_dataset, num_epochs, optimizer, loss_func, batch_size, device):
    kf = KFold(n_splits=num_epochs)
    for epoch, (train_split_idx, val_split_idx) in enumerate(kf.split(range(len(train_dataset)))):
        fold_train_subset = Subset(train_dataset, train_split_idx)
        fold_val_subset = Subset(train_dataset, val_split_idx)

        epoch_train_loader = DataLoader(fold_train_subset, batch_size=batch_size, shuffle=True)
        epoch_val_loader = DataLoader(fold_val_subset, batch_size=batch_size)

        epoch_loss = train_epoch(model, epoch_train_loader, optimizer, loss_func, device)
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {np.mean(epoch_loss):.4f}')
        validate(model, loss_func, epoch_val_loader, device)


def quantize_model(model):
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8  # Quantize Linear layers to int8
    )
    return quantized_model


def create_sliding_window_mask(seq_len, window_size):
    """
    Creates a mask for a sequence where each position attends to itself and
    'window_size' positions before and after it.

    Args:
        seq_len (int): Length of the sequence.
        window_size (int): Number of positions before and after to attend to.

    Returns:
        torch.Tensor: The attention mask of shape (seq_len, seq_len).
    """
    # Initialize the mask with zeros
    mask = torch.zeros(seq_len, seq_len)

    # Apply the sliding window constraint
    for i in range(seq_len):
        # Mask out positions outside the window [i - window_size, i + window_size]
        if i - window_size > 0:
            mask[i, :i - window_size] = float('-inf')
        if i + window_size + 1 < seq_len:
            mask[i, i + window_size + 1:] = float('-inf')

    return mask
