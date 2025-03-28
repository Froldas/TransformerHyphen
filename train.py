import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

import src.utils as utils
from src.ConfDict import Models, Encodings
from src.dataset import HyphenationDataset

YML_CONF_PATH = "configuration.yml"


def main():
    if len(sys.argv) > 1:
        config = utils.load_yaml_conf(Path(sys.argv[1]))
    else:
        config = utils.load_yaml_conf(Path(YML_CONF_PATH))

    os.makedirs(config["work_dir"], exist_ok=True)

    utils.setup_logger(Path(config["work_dir"]) / config["training_log_path"])
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # set seed for reproducibility
    utils.set_seed(config["seed"])

    # Create datasets and dataloaders
    dataset = HyphenationDataset(data_file=config["dataset"],
                                 work_dir=config["work_dir"],
                                 encoding=Encodings().encodings[config["encoding"]],
                                 print_info=config["print_dataset_statistics"])

    train_dataset, eval_dataset = utils.split_dataset(dataset, config["train_split"])

    model = Models(dataset.num_input_tokens,
                   dataset.encoding_size,
                   dataset.output_size).models[config["model"]].to(device)

    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training Loop
    num_epochs = config["num_epochs"]
    kf = KFold(n_splits=num_epochs)

    for epoch, (train_split_idx, val_split_idx) in enumerate(kf.split(range(len(train_dataset)))):
        fold_train_subset = Subset(train_dataset, train_split_idx)
        fold_val_subset = Subset(train_dataset, val_split_idx)

        epoch_train_loader = DataLoader(fold_train_subset, batch_size=config["batch_size"], shuffle=True)
        epoch_val_loader = DataLoader(fold_val_subset, batch_size=config["batch_size"])

        epoch_loss = utils.train_epoch(model, epoch_train_loader, optimizer, loss_func, device)
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {np.mean(epoch_loss):.4f}')
        utils.validate(model, loss_func, epoch_val_loader, device)

    quantized_model = utils.quantize_model(model)

    utils.setup_logger(Path(config["work_dir"]) / "eval_metrics.log")

    X = []
    y = []
    for data_point in eval_dataset:
        features, label = data_point
        X.append(features)  # Convert features to NumPy array
        y.append(label)

    utils.model_evaluation(model, X, y, config["dataset"], label="Original model")
    utils.model_evaluation(quantized_model, X, y, config["dataset"], label="Quantized model")

    utils.save_model(model, Path(config["work_dir"]) / config["model_path"])
    utils.save_model(quantized_model, Path(config["work_dir"]) / ("quant_" + config["model_path"]))

    utils.visualize(model, dataset, config["work_dir"])


if __name__ == "__main__":
    main()
