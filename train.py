import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader

from src.dataset import HyphenationDataset
from src.models.simple_mlp import SimpleMLP
from src.utils import load_yaml_conf

YML_CONF_PATH = "configuration.yml"

def main():
    config = load_yaml_conf(Path(YML_CONF_PATH))
    # Create datasets and dataloaders
    dataset = HyphenationDataset(data_file=config["dataset"],
                                 work_dir=config["work_dir"],
                                 print_info=config["print_dataset_statistics"])
    train_size = int(config["train_split"] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleMLP(dataset.input_size, 64, dataset.output_size).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training Loop
    num_epochs = config["num_epochs"]
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss))
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {np.mean(epoch_loss):.4f}')
        model.eval()
        with torch.no_grad():
            val_loss = []
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss.append(float(loss))
            print(f'Val loss: {np.mean(val_loss):.4f}')

    # Save the model
    os.makedirs(config["work_dir"], exist_ok=True)
    output_model_path = Path(config["work_dir"]) / config["model_path"]
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")


if __name__ == "__main__":
    main()
