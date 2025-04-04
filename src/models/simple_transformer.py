import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import create_sliding_window_mask

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size, bias=False)
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, x, mask=None):
        batch_size, seq_length, embed_size = x.shape
        assert embed_size == self.embed_size, "Input embedding size must match model embedding size"

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        energy = torch.matmul(Q, K.transpose(-2, -1)) / (embed_size ** 0.5)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, V)
        return out


class SimpleTransformer(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size):
        super(SimpleTransformer, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.attention = SelfAttention(embed_size)
        self.fc_in = nn.Linear(input_tokens * embed_size, hidden_size)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.shape[0], self.input_tokens, -1)
        x = self.attention(x)
        x = x.view(-1, self.input_tokens * self.embed_size)
        #x = x.mean(dim=1)
        x = F.relu(self.fc_in(x))
        x = F.relu( self.fc_hidden(x))
        x = self.fc_out(x)
        x = F.sigmoid(x)
        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > 0.7).float()
        return x


class SimpleTransformerMasked(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size):
        super(SimpleTransformerMasked, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.attention = SelfAttention(embed_size)
        self.fc_in = nn.Linear(input_tokens * embed_size, hidden_size)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.shape[0], self.input_tokens, -1)
        x = self.attention(x, mask=create_sliding_window_mask(self.input_tokens, 5))
        x = x.view(-1, self.input_tokens * self.embed_size)
        #x = x.mean(dim=1)
        x = F.relu(self.fc_in(x))
        x = F.relu(self.fc_hidden(x))
        x = self.fc_out(x)
        x = F.sigmoid(x)
        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > 0.7).float()
        return x


class SimpleTransformerResidual(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size):
        super(SimpleTransformerResidual, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.attention = SelfAttention(embed_size)
        self.fc_in = nn.Linear(input_tokens * embed_size, hidden_size)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        residual = x.view(x.shape[0], -1)
        x = x.view(x.shape[0], self.input_tokens, -1)
        x = self.attention(x)
        x = x.view(-1, self.input_tokens * self.embed_size)
        x += residual
        #x = x.mean(dim=1)
        x = F.relu(self.fc_in(x))
        x = F.relu(self.fc_hidden(x))
        x = self.fc_out(x)
        x = F.sigmoid(x)
        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > 0.5).float()
        return x


class SimpleTransformerResidualNormalized(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size):
        super(SimpleTransformerResidualNormalized, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.attention = SelfAttention(embed_size)
        self.fc_in = nn.Linear(input_tokens * embed_size, hidden_size)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        residual = x.view(x.shape[0], -1)
        x = x.view(x.shape[0], self.input_tokens, -1)
        x = self.attention(x)
        x = x.view(-1, self.input_tokens * self.embed_size)
        x += residual
        # x = x.mean(dim=1)
        x = F.relu(self.fc_in(x))
        x = self.fc_hidden(x)
        x = F.relu(F.normalize(x))
        x = self.fc_out(x)
        x = F.sigmoid(x)
        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > 0.5).float()
        return x


class SimpleReverseTransformerResidualNormalized(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size):
        super(SimpleReverseTransformerResidualNormalized, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.attention = SelfAttention(1)
        self.fc_in = nn.Linear(input_tokens * embed_size, hidden_size)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.shape[0], self.input_tokens * self.embed_size)
        x = F.relu(self.fc_in(x))
        residual = x
        x = x.view(x.shape[0], self.hidden_size, 1)
        x = self.attention(x)
        x = x.view(-1, self.hidden_size)
        x += residual
        residual = x
        x = self.fc_hidden(x)
        x += residual
        x = F.relu(F.normalize(x))
        x = self.fc_out(x)
        x = F.sigmoid(x)
        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > 0.2).float()
        return x

class SimpleReverseTransformer(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size):
        super(SimpleReverseTransformer, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.attention = SelfAttention(1)
        self.fc_in = nn.Linear(input_tokens * embed_size, hidden_size)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.shape[0], self.input_tokens * self.embed_size)
        x = F.relu(self.fc_in(x))
        x = x.view(x.shape[0], self.hidden_size, 1)
        x = self.attention(x)
        x = x.view(-1, self.hidden_size)
        x = F.relu(self.fc_hidden(x))
        x = self.fc_out(x)
        x = F.sigmoid(x)
        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > 0.5).float()
        return x

class SimpleTransformerResidualDeep(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size):
        super(SimpleTransformerResidualDeep, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
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
            x = (x > 0.8).float()
        return x
