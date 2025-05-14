import torch.nn as nn

from src.models.modules import FeedForward, Attention


class TransformerCombined1(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size, hyphen_threshold=0.5):
        super(TransformerCombined1, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.hyphen_threshold = hyphen_threshold
        self.attention = Attention(embed_size, residual=True, normalization='batchnorm')
        self.fc_in = FeedForward(input_tokens * embed_size, hidden_size, normalization='batchnorm')
        self.fc_hidden1 = FeedForward(hidden_size, hidden_size, normalization='batchnorm')
        self.fc_hidden2 = FeedForward(hidden_size, hidden_size, normalization='batchnorm')
        self.fc_hidden3 = FeedForward(hidden_size, hidden_size, normalization='batchnorm')
        self.fc_out = FeedForward(hidden_size, output_size, activation="sigmoid")

    def forward(self, x):
        x = x.reshape(-1, self.input_tokens, self.embed_size)
        x = self.attention(x)
        x = x.reshape(-1, self.input_tokens * self.embed_size)
        x = self.fc_in(x)
        res = x
        x = self.fc_hidden1(x)
        x = self.fc_hidden2(x)
        x = self.fc_hidden3(x)
        x = res + x
        x = self.fc_out(x)

        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > self.hyphen_threshold).float()
        return x

class AdvancedTransformerResidualDeep(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size, hyphen_threshold=0.5):
        super(AdvancedTransformerResidualDeep, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.hyphen_threshold = hyphen_threshold
        self.attention = Attention(embed_size, residual=True, normalization='layernorm')
        self.fc_hidden1 = FeedForward(input_tokens * embed_size, hidden_size, residual=True, normalization='layernorm')
        self.fc_hidden2 = FeedForward(hidden_size, hidden_size, residual=True, normalization='layernorm')
        self.fc_out = FeedForward(hidden_size, output_size, activation="sigmoid")

    def forward(self, x):
        x = x.view(-1, self.input_tokens, self.embed_size)
        x = self.attention(x)
        x = x.view(-1, self.input_tokens*self.embed_size)
        x = self.fc_hidden1(x)
        x = self.fc_hidden2(x)
        x = self.fc_out(x)

        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > self.hyphen_threshold).float()
        return x

class AdvancedTransformerResidualDeepMHead(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size, hyphen_threshold=0.5):
        super(AdvancedTransformerResidualDeepMHead, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.hyphen_threshold = hyphen_threshold
        self.attention = Attention(embed_size, residual=True, normalization='layernorm')
        #self.attention = Attention(embed_size, residual=True, normalization='layernorm', num_heads=8)
        self.fc_hidden1 = FeedForward(input_tokens * embed_size, hidden_size, residual=True, normalization='layernorm')
        self.fc_hidden2 = FeedForward(hidden_size, hidden_size, residual=True, normalization='layernorm')
        self.fc_out = FeedForward(hidden_size, output_size, activation="sigmoid")

    def forward(self, x):
        x = x.view(-1, self.input_tokens, self.embed_size)
        x = self.attention(x)
        x = x.view(-1, self.input_tokens*self.embed_size)
        x = self.fc_hidden1(x)
        x = self.fc_hidden2(x)
        x = self.fc_out(x)

        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > self.hyphen_threshold).float()
        return x

class TransformerCombined2(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size, hyphen_threshold=0.5):
        super(TransformerCombined2, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.hyphen_threshold = hyphen_threshold

        self.attention1 = Attention(embed_size, normalization='batchnorm')
        self.fc_att1 = FeedForward(input_tokens * embed_size, input_tokens * embed_size, normalization='batchnorm')
        self.attention2 = Attention(embed_size, normalization='batchnorm')
        self.fc_att2 = FeedForward(input_tokens * embed_size, input_tokens * embed_size, normalization='batchnorm')
        self.attention3 = Attention(embed_size, residual=True, normalization='batchnorm')

        self.fc_in = FeedForward(input_tokens * embed_size, hidden_size, normalization='batchnorm')
        self.fc_out = FeedForward(hidden_size, output_size, activation="sigmoid")

    def forward(self, x):
        res = x
        x = x.reshape(-1, self.input_tokens, self.embed_size)
        x = self.attention1(x)
        x = x.reshape(-1, self.input_tokens * self.embed_size)
        x = self.fc_att1(x)
        x += res

        res = x
        x = x.reshape(-1, self.input_tokens, self.embed_size)
        x = self.attention2(x)
        x = x.reshape(-1, self.input_tokens * self.embed_size)
        x = self.fc_att2(x)
        x += res

        x = x.reshape(-1, self.input_tokens, self.embed_size)
        x = self.attention3(x)
        x = x.reshape(-1, self.input_tokens * self.embed_size)
        x = self.fc_in(x)
        x = self.fc_out(x)

        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > self.hyphen_threshold).float()
        return x


class TransformerCombined3(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size, hyphen_threshold=0.5):
        super(TransformerCombined3, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.hyphen_threshold = hyphen_threshold
        self.attention1 = Attention(embed_size, residual=True, normalization='batchnorm')
        self.attention2 = Attention(embed_size, residual=True, normalization='batchnorm')
        self.fc_in = FeedForward(input_tokens * embed_size, hidden_size, normalization='batchnorm')
        self.fc_hidden1 = FeedForward(hidden_size, hidden_size, normalization='batchnorm', residual=True)
        self.fc_out = FeedForward(hidden_size, output_size, activation="sigmoid")

    def forward(self, x):
        x = x.reshape(-1, self.input_tokens, self.embed_size)
        x = self.attention1(x)
        x = self.attention2(x)
        x = x.reshape(-1, self.input_tokens * self.embed_size)
        x = self.fc_in(x)
        x = self.fc_hidden1(x)
        x = self.fc_out(x)

        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > self.hyphen_threshold).float()
        return x


