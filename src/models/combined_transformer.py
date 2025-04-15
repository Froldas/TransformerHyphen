import torch.nn as nn

from src.models.modules import FeedForward, SelfAttention


class TransformerCombined1(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size):
        super(TransformerCombined1, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.attention = SelfAttention(embed_size, residual=True, normalization='batchnorm')
        self.fc_in = FeedForward(input_tokens * embed_size, hidden_size, normalization='batchnorm')
        self.fc_hidden1 = FeedForward(hidden_size, hidden_size, normalization='batchnorm')
        self.fc_hidden2 = FeedForward(hidden_size, hidden_size, normalization='batchnorm')
        self.fc_hidden3 = FeedForward(hidden_size, hidden_size, normalization='batchnorm')
        self.fc_out = FeedForward(hidden_size, output_size, activation="sigmoid")

    def forward(self, x):
        x = x.view(-1, self.input_tokens, self.embed_size)
        x = self.attention(x)
        x = x.view(-1, self.input_tokens * self.embed_size)
        x = self.fc_in(x)
        res = x
        x = self.fc_hidden1(x)
        x = self.fc_hidden2(x)
        x = self.fc_hidden3(x)
        x = res + x
        x = self.fc_out(x)

        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > 0.7).float()
        return x
