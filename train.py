import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader
from torchview import draw_graph

from src.dataset import HyphenationDataset
from src.ConfDict import Models, Encodings
from src.utils import set_seed, load_yaml_conf, train_epoch, validate, setup_logger

YML_CONF_PATH = "configuration.yml"


def main():
    if len(sys.argv) > 1:
        config = load_yaml_conf(Path(sys.argv[1]))
    else:
        config = load_yaml_conf(Path(YML_CONF_PATH))

    os.makedirs(config["work_dir"], exist_ok=True)

    setup_logger(Path(config["work_dir"]) / config["training_log_path"])
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # set seed for reproducibility
    set_seed(config["seed"])

    # Create datasets and dataloaders
    dataset = HyphenationDataset(data_file=config["dataset"],
                                 work_dir=config["work_dir"],
                                 encoding=Encodings().encodings[config["encoding"]],
                                 print_info=config["print_dataset_statistics"])

    train_size = int(config["train_split"] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    model = Models(dataset.num_input_tokens, dataset.encoding_size, dataset.output_size).models[config["model"]].to(
        device)

    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training Loop
    num_epochs = config["num_epochs"]

    for epoch in range(num_epochs):
        epoch_loss = train_epoch(model, train_loader, optimizer, loss_func, device)
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {np.mean(epoch_loss):.4f}')
        validate(model, loss_func, val_loader, device)

    # Save the model
    os.makedirs(config["work_dir"], exist_ok=True)
    output_model_path = Path(config["work_dir"]) / config["model_path"]
    torch.save(model.state_dict(), output_model_path)
    logging.info(f"Model saved to {output_model_path}")

    # visualization of the architecture
    model_graph = draw_graph(model, input_size=(1, dataset.input_size), expand_nested=True)
    model_graph.visual_graph.render(filename="model", format='pdf', directory=config["work_dir"])


if __name__ == "__main__":
    main()
