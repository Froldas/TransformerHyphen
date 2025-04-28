import torch.nn as nn

from src.models.modules import FeedForward, SelfAttention, Conv1DBlock


class TransformerCombined1(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size, hyphen_threshold=0.5):
        super(TransformerCombined1, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.hyphen_threshold = hyphen_threshold
        self.attention = SelfAttention(embed_size, residual=True, normalization='batchnorm')
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


class TransformerCombined2(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size, hyphen_threshold=0.5):
        super(TransformerCombined2, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.hyphen_threshold = hyphen_threshold
        self.attention1 = SelfAttention(embed_size, residual=True, normalization='batchnorm')
        self.attention2 = SelfAttention(embed_size, residual=True, normalization='batchnorm')
        self.attention3 = SelfAttention(embed_size, residual=True, normalization='batchnorm')
        self.fc_in = FeedForward(input_tokens * embed_size, hidden_size, normalization='batchnorm')
        self.fc_hidden1 = FeedForward(hidden_size, hidden_size, normalization='batchnorm', residual=True)
        self.fc_out = FeedForward(hidden_size, output_size, activation="sigmoid")

    def forward(self, x):
        x = x.reshape(-1, self.input_tokens, self.embed_size)
        x = self.attention1(x)
        x = self.attention2(x)
        x = self.attention3(x)
        x = x.reshape(-1, self.input_tokens * self.embed_size)
        x = self.fc_in(x)
        x = self.fc_hidden1(x)
        x = self.fc_out(x)

        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > self.hyphen_threshold).float()
        return x


class TransformerCombined3(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size, hyphen_threshold=0.5, kernel_count=8):
        super(TransformerCombined3, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.hyphen_threshold = hyphen_threshold
        self.kernel_count = kernel_count
        self.conv = Conv1DBlock(1, self.kernel_count, embed_size * 7, stride=embed_size,
                                padding=(embed_size * 7 - 1) // 2, residual=True, normalization='batchnorm')
        self.attention1 = SelfAttention(kernel_count, residual=True, normalization='batchnorm')
        self.attention2 = SelfAttention(kernel_count, residual=True, normalization='batchnorm')
        self.attention3 = SelfAttention(kernel_count, residual=True, normalization='batchnorm')
        self.fc_in = FeedForward(input_tokens * kernel_count, hidden_size, normalization='batchnorm')
        self.fc_hidden1 = FeedForward(hidden_size, hidden_size, normalization='batchnorm', residual=True)
        self.fc_out = FeedForward(hidden_size, output_size, activation="sigmoid")

    def forward(self, x):
        x = x.view(-1, 1, self.embed_size * self.input_tokens)
        x = self.conv(x)
        x = x.reshape(-1, self.input_tokens, self.kernel_count)
        x = self.attention1(x)
        x = self.attention2(x)
        x = self.attention3(x)
        x = x.reshape(-1, self.input_tokens * self.kernel_count)
        x = self.fc_in(x)
        x = self.fc_hidden1(x)
        x = self.fc_out(x)

        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > self.hyphen_threshold).float()
        return x


class TransformerCombined4(nn.Module):
    def __init__(self, input_tokens, embed_size, hidden_size, output_size, hyphen_threshold=0.5):
        super(TransformerCombined4, self).__init__()
        self.input_tokens = input_tokens
        self.embed_size = embed_size
        self.hyphen_threshold = hyphen_threshold

        self.attention1 = SelfAttention(embed_size, residual=True, normalization='batchnorm')
        self.fc_att1 = FeedForward(input_tokens * embed_size, input_tokens * embed_size, normalization='batchnorm')
        self.attention2 = SelfAttention(embed_size, residual=True, normalization='batchnorm')
        self.fc_att2 = FeedForward(input_tokens * embed_size, input_tokens * embed_size, normalization='batchnorm')
        self.attention3 = SelfAttention(embed_size, residual=True, normalization='batchnorm')

        self.fc_in = FeedForward(input_tokens * embed_size, hidden_size, normalization='batchnorm')
        self.fc_out = FeedForward(hidden_size, output_size, activation="sigmoid")

    def forward(self, x):
        x = x.reshape(-1, self.input_tokens, self.embed_size)
        x = self.attention1(x)
        x = x.reshape(-1, self.input_tokens * self.embed_size)
        x = self.fc_att1(x)
        x = x.reshape(-1, self.input_tokens, self.embed_size)
        x = self.attention2(x)
        x = x.reshape(-1, self.input_tokens * self.embed_size)
        x = self.fc_att2(x)
        x = x.reshape(-1, self.input_tokens, self.embed_size)
        x = self.attention3(x)
        x = x.reshape(-1, self.input_tokens * self.embed_size)
        x = self.fc_in(x)
        x = self.fc_out(x)

        if not self.training:
            # return 1 or 0 based on a threshold
            x = (x > self.hyphen_threshold).float()
        return x
