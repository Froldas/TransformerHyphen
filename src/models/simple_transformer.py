import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, embed_size, num_classes):
        super(SimpleTransformer, self).__init__()
        self.attention = SelfAttention(embed_size)
        self.fc = nn.Linear(168, 32)
        self.fc_out = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], 28, -1)
        x = self.attention(x)
        x = x.view(-1, 168)
        #x = x.mean(dim=1)
        x = F.relu(self.fc(x))
        x = self.fc_out(x)
        x = F.sigmoid(x)
        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > 0.7).float()
        return x
