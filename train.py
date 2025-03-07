
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import HyphenationDataset, insert_hyphenation
from src.model import SimpleMLP
from torch.utils.data import DataLoader

batch_size = 8
data_file = "data/cs-all-cstenten.wlh"

def main():
    # Create datasets and dataloaders
    dataset = HyphenationDataset(data_file=data_file)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Check if CUDA is available
    device = "cuda" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleMLP(dataset.input_size,512, dataset.output_size).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    num_epochs = 10
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
    torch.save(model.state_dict(), 'simple_mlp_model.pth')
    print("Model saved to simple_mlp_model.pth")

    X = []
    y = []
    for data_point in dataset:
        features, label = data_point
        X.append(features)  # Convert features to NumPy array
        y.append(label)
    y_pred = model(torch.Tensor(np.array(X)).to(device))
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(torch.Tensor(np.array(y)).detach().numpy(), y_pred.to("cpu").detach().numpy())

    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()