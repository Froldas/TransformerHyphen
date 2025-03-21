import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from src.dataset import HyphenationDataset
from src.ConfDict import Models, Encodings
import src.utils as utils

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

    train_loader, val_loader = utils.split_dataset(dataset, config["train_split"], config["batch_size"])

    model = Models(dataset.num_input_tokens, dataset.encoding_size, dataset.output_size).models[config["model"]].to(
        device)

    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training Loop
    num_epochs = config["num_epochs"]

    for epoch in range(num_epochs):
        epoch_loss = utils.train_epoch(model, train_loader, optimizer, loss_func, device)
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {np.mean(epoch_loss):.4f}')
        utils.validate(model, loss_func, val_loader, device)

    utils.save_model(model, Path(config["work_dir"]) / config["model_path"])
    utils.visualize(model, dataset, config["work_dir"])


if __name__ == "__main__":
    main()
