import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads=1, dropout=0.0,
                 residual=False, normalization=None):
        super(Attention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.residual = residual
        self.dropout_ratio = dropout

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if self.dropout_ratio > 0.0:
            self.dropout = nn.Dropout(dropout)

        # Normalization layer
        if normalization == "layernorm":
            self.norm = nn.LayerNorm(embed_dim)
        elif normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(embed_dim)
        elif normalization is None:
            self.norm = None
        else:
            raise ValueError("Unsupported normalization. Choose 'layernorm', 'batchnorm', or None.")

        # make the weights class property so it can be accessed for visualization
        self.attn_weights = None

    def forward(self, x, mask=None):
        # x: (batch_size, seq_length, embed_dim)
        residual = x

        batch_size, seq_length, _ = x.size()

        # Linear projections + reshape for multi-head
        Q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        self.attn_weights = F.softmax(scores, dim=-1)
        if self.dropout_ratio > 0.0:
            self.attn_weights = self.dropout(self.attn_weights)

        attn_output = torch.matmul(self.attn_weights, V)  # (batch, heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        output = self.out_proj(attn_output)
        if self.dropout_ratio > 0.0:
            output = self.dropout(output)

        if self.residual:
            output = output + residual  # Add residual connection

        if self.norm:
            if isinstance(self.norm, nn.BatchNorm1d):
                output = output.transpose(1, 2)  # (batch, embed_dim, seq_len)
                output = self.norm(output)
                output = output.transpose(1, 2)
            else:
                output = self.norm(output)

        return output


class FeedForward(nn.Module):
    def __init__(self, input_dim, output_dim=None, activation='relu',
                 normalization=None, residual=False, dropout=0.0):
        super(FeedForward, self).__init__()

        self.output_dim = output_dim or input_dim
        self.residual = residual and (self.output_dim == input_dim)
        self.dropout_val = dropout

        # Linear layer
        self.linear = nn.Linear(input_dim, self.output_dim)

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation. Choose 'relu' or 'sigmoid'.")

        # Normalization
        if normalization == 'layernorm':
            self.norm = nn.LayerNorm(self.output_dim)
        elif normalization == 'batchnorm':
            self.norm = nn.BatchNorm1d(self.output_dim)
        elif normalization is None:
            self.norm = None
        else:
            raise ValueError("Unsupported normalization. Choose 'layernorm', 'batchnorm', or None.")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)

        if self.dropout_val > 0.0:
            out = self.dropout(out)

        if self.residual:
            out = out + x  # Residual connection

        if self.norm:
            if isinstance(self.norm, nn.BatchNorm1d):
                out = self.norm(out)
            else:
                out = self.norm(out)

        return out

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, activation='relu', normalization=None,
                 residual=False, dropout=0.0):
        super(Conv1DBlock, self).__init__()

        # Determine padding to maintain sequence length if not specified
        if padding is None:
            padding = (kernel_size - 1) // 2

        self.residual = residual and (in_channels == out_channels)

        # Convolutional layer
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding)

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation. Choose 'relu' or 'sigmoid'.")

        # Normalization layer
        if normalization == 'layernorm':
            self.norm = nn.LayerNorm(out_channels)
        elif normalization == 'batchnorm':
            self.norm = nn.BatchNorm1d(out_channels)
        elif normalization is None:
            self.norm = None
        else:
            raise ValueError("Unsupported normalization. Choose 'layernorm', 'batchnorm', or None.")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, in_channels, seq_len)
        """
        out = self.conv(x)  # Shape: (batch_size, out_channels, seq_len)

        if self.norm:
            if isinstance(self.norm, nn.LayerNorm):
                # Transpose to (batch_size, seq_len, out_channels) for LayerNorm
                out = out.transpose(1, 2)
                out = self.norm(out)
                out = out.transpose(1, 2)
            else:
                out = self.norm(out)

        out = self.activation(out)
        out = self.dropout(out)

        if self.residual:
            out = out + x  # Residual connection

        return out