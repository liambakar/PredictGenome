import torch
import torch.nn as nn
import numpy as np


class HallmarkFeatureProcessor(nn.Module):
    def __init__(self, output_embedding_dim,
                 cnn_filters=32, cnn_kernel_size=3):
        """
        Processes a single hallmark's variable-length feature vector to a fixed embedding.
        Input shape during forward: (batch_size, 1, L_i) where L_i can vary.
        Output shape: (batch_size, output_embedding_dim)
        """
        super(HallmarkFeatureProcessor, self).__init__()

        if cnn_kernel_size % 2 == 0:
            raise ValueError(
                "cnn_kernel_size must be odd for simple 'same' padding calculation.")

        # Convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=cnn_filters,
                               kernel_size=cnn_kernel_size,
                               # 'same' padding
                               padding=(cnn_kernel_size - 1) // 2)
        self.bn_conv1 = nn.BatchNorm1d(cnn_filters)

        # Adaptive Max Pooling: crucial for handling variable L_i and getting fixed output size
        self.adaptive_pool = nn.AdaptiveMaxPool1d(
            1)  # Outputs (batch_size, cnn_filters, 1)

        # Fully connected layer to get the final desired embedding dimension
        self.fc_embed = nn.Linear(cnn_filters, output_embedding_dim)
        self.bn_embed = nn.BatchNorm1d(output_embedding_dim)

    def forward(self, x_li):
        # x_li shape: (batch_size, 1, L_i) - L_i is variable for this hallmark

        h = self.conv1(x_li)
        h = self.bn_conv1(h)
        h = nn.functional.relu(h)

        h = self.adaptive_pool(h)  # Shape: (batch_size, cnn_filters, 1)
        h = h.squeeze(2)           # Shape: (batch_size, cnn_filters)

        embedding = self.fc_embed(h)
        embedding = self.bn_embed(embedding)
        # Shape: (batch_size, output_embedding_dim)
        embedding = nn.functional.relu(embedding)

        return embedding


class PositionalEncoding(nn.Module):
    """
    Simple learned positional encoding.
    """

    def __init__(self, d_model, max_len):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        # x is (batch_size, seq_len, d_model)
        # Use only up to x.size(1) positions
        return x + self.position_embedding[:x.size(1), :].unsqueeze(0)
