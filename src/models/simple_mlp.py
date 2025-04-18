import torch.nn as nn
from src.models.modules import FeedForward


class SimpleMLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, hyphen_threshold=0.5):
        super(SimpleMLP, self).__init__()
        self.input_size = input_size
        self.hyphen_threshold = hyphen_threshold
        self.fc1 = FeedForward(input_size, hidden_size)
        self.fc2 = FeedForward(hidden_size, hidden_size)
        self.fc3 = FeedForward(hidden_size, output_size, activation="sigmoid")

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > self.hyphen_threshold).float()

        return x
