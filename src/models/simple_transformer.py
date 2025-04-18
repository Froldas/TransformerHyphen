import torch.nn as nn
import torch.nn.functional as F

from src.models.modules import FeedForward, SelfAttention
from src.utils import create_sliding_window_mask


class SimpleTransformer(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size, hyphen_threshold=0.5):
        super(SimpleTransformer, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.hyphen_threshold = hyphen_threshold
        self.attention = SelfAttention(embed_size)
        self.fc_in = FeedForward(input_tokens * embed_size, hidden_size)
        self.fc_hidden = FeedForward(hidden_size, hidden_size)
        self.fc_out = FeedForward(hidden_size, output_size, activation="sigmoid")

    def forward(self, x):
        x = x.view(-1, self.input_tokens, self.embed_size)
        x = self.attention(x)
        x = x.view(-1, self.input_tokens*self.embed_size)
        x = self.fc_in(x)
        x = self.fc_hidden(x)
        x = self.fc_out(x)

        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > self.hyphen_threshold).float()
        return x


class SimpleTransformerResidual(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size, hyphen_threshold=0.5):
        super(SimpleTransformerResidual, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.hyphen_threshold = hyphen_threshold
        self.attention = SelfAttention(embed_size)
        self.fc_in = FeedForward(input_tokens * embed_size, hidden_size, residual=True, normalization='layernorm')
        self.fc_hidden = FeedForward(hidden_size, hidden_size, residual=True, normalization='layernorm')
        self.fc_out = FeedForward(hidden_size, output_size, activation="sigmoid")

    def forward(self, x):
        x = x.view(-1, self.input_tokens, self.embed_size)
        x = self.attention(x)
        x = x.view(-1, self.input_tokens*self.embed_size)
        x = self.fc_in(x)
        x = self.fc_hidden(x)
        x = self.fc_out(x)

        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > self.hyphen_threshold).float()
        return x


class SimpleTransformerMaskWindow(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size, hyphen_threshold=0.5):
        super(SimpleTransformerMaskWindow, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.hyphen_threshold = hyphen_threshold
        self.attention = SelfAttention(embed_size)
        self.fc_in = FeedForward(input_tokens * embed_size, hidden_size)
        self.fc_hidden = FeedForward(hidden_size, hidden_size)
        self.fc_out = FeedForward(hidden_size, output_size, activation="sigmoid")

    def forward(self, x):
        x = x.view(-1, self.input_tokens, self.embed_size)
        x = self.attention(x, mask=create_sliding_window_mask(self.input_tokens, 5))
        x = x.view(-1, self.input_tokens*self.embed_size)
        x = self.fc_in(x)
        x = self.fc_hidden(x)
        x = self.fc_out(x)

        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > self.hyphen_threshold).float()
        return x


class SimpleTransformerReversed(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size, hyphen_threshold=0.5):
        super(SimpleTransformerReversed, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.hyphen_threshold = hyphen_threshold
        self.attention = SelfAttention(embed_size)
        self.fc_in = FeedForward(input_tokens * embed_size, hidden_size)
        self.fc_hidden = FeedForward(hidden_size, hidden_size)
        self.fc_out = FeedForward(hidden_size, output_size, activation="sigmoid")

    def forward(self, x):
        x = self.fc_in(x)
        x = x.view(-1, self.hidden_size, self.embed_size)
        x = self.attention(x)
        x = x.view(-1, self.hidden_size * self.embed_size)
        x = self.fc_hidden(x)
        x = self.fc_out(x)

        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > self.hyphen_threshold).float()
        return x


class SimpleTransformerResidualDeep(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size, hyphen_threshold=0.5):
        super(SimpleTransformerResidualDeep, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.hyphen_threshold = hyphen_threshold
        self.attention = SelfAttention(embed_size)
        self.fc_in = nn.Linear(input_tokens * embed_size, hidden_size)
        self.fc_hidden1 = nn.Linear(hidden_size, hidden_size)
        self.fc_hidden2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        residual = x.view(x.shape[0], -1)
        x = x.view(x.shape[0], self.input_tokens, -1)
        x = self.attention(x)
        x = x.view(-1, self.input_tokens * self.embed_size)
        x += residual
        # x = x.mean(dim=1)
        x = F.relu(self.fc_in(x))
        x = F.relu(self.fc_hidden1(x))
        x = F.relu(self.fc_hidden2(x))
        x = self.fc_out(x)
        x = F.sigmoid(x)
        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > self.hyphen_threshold).float()
        return x
