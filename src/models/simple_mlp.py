import torch.nn as nn
from src.models.modules import FeedForward, Conv1DBlock


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

class SimpleMLPConvolution(nn.Module):

    def __init__(self, input_tokens, embed_size, hidden_size, output_size, hyphen_threshold=0.5, kernel_count=16):
        super(SimpleMLPConvolution, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.hyphen_threshold = hyphen_threshold
        self.kernel_count = kernel_count
        self.conv = Conv1DBlock(1, self.kernel_count, embed_size * 7, stride=embed_size,
                                 padding=(embed_size * 7 - 1) // 2)
        self.fc1 = FeedForward(input_tokens * self.kernel_count, hidden_size)
        self.fc2 = FeedForward(hidden_size, hidden_size)
        self.fc3 = FeedForward(hidden_size, output_size, activation="sigmoid")

    def forward(self, x):
        x = x.view(-1, 1, self.embed_size * self.input_tokens)
        x = self.conv(x)
        x = x.view(-1, self.input_tokens * self.kernel_count)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > self.hyphen_threshold).float()

        return x
