import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score
from torch.utils.data import DataLoader, Subset

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

    # Apply Dynamic Quantization (Converts Linear Layers to int8)
    quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8  # Quantize Linear layers to int8
    )

    X = []
    y = []
    for data_point in eval_dataset:
        features, label = data_point
        X.append(features)  # Convert features to NumPy array
        y.append(label)

    full_x_pred = model(torch.Tensor(np.array(X)).to("cpu"))
    quant_x_pred = quantized_model(torch.Tensor(np.array(X)).to("cpu"))

    full_accuracy = accuracy_score(torch.Tensor(np.array(y)).detach().numpy(),
                                   full_x_pred.to("cpu").detach().numpy())
    full_recall = recall_score(torch.Tensor(np.array(y)).detach().numpy(), full_x_pred.to("cpu").detach().numpy(),
                               average="samples")
    quant_accuracy = accuracy_score(torch.Tensor(np.array(y)).detach().numpy(),
                                    quant_x_pred.to("cpu").detach().numpy())
    quant_recall = recall_score(torch.Tensor(np.array(y)).detach().numpy(),
                                quant_x_pred.to("cpu").detach().numpy(),
                                average="samples")

    utils.save_model(quantized_model, Path(config["work_dir"]) / config["model_path"])
    utils.visualize(model, dataset, config["work_dir"])
    utils.save_model(model, Path(config["work_dir"]) / config["model_path"])
    utils.visualize(model, dataset, config["work_dir"])

    with open(Path(config["work_dir"]) / "eval_metrics.log", "w+", encoding="utf-8") as f:
        f.writelines(f"Metrics on unseen data:\n")
        print(f"Metrics on unseen data:\n")
        f.writelines(f"    FULL Accuracy: {full_accuracy:.4f}\n")
        print(f"    FULL Accuracy: {full_accuracy:.4f}")
        f.writelines(f"    FULL Recall: {full_recall:.4f}\n")
        print(f"    FULL Recall: {full_recall:.4f}")
        f.writelines(f"    QUANTIZED Accuracy: {quant_accuracy:.4f}\n")
        print(f"    QUANTIZED Accuracy: {quant_accuracy:.4f}")
        f.writelines(f"    QUANTIZED Recall: {quant_recall:.4f}\n")
        print(f"    QUANTIZED Recall: {quant_recall:.4f}")

    def model_size(model):
        torch.save(model.state_dict(), "temp.pth")
        size = os.path.getsize("temp.pth") / 1024  # Convert to KB
        os.remove("temp.pth")
        return size

    print(f"Original Model Size: {model_size(model):.2f} KB")
    print(f"Quantized Model Size: {model_size(quantized_model):.2f} KB")


if __name__ == "__main__":
    main()
