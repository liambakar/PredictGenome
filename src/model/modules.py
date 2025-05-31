import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Adds positional information to input embeddings.
    Ensures compatibility with batch_first=True convention.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(
            0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            if d_model > 1:
                pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CoAttentionModule(nn.Module):
    """
    Implements the Co-Attention mechanism between the input sequence and learnable queries.
    Assumes batch_first=True for all MultiheadAttention layers.
    """

    def __init__(self, embed_dim, num_heads, num_queries, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_queries = num_queries

        self.coattention_queries = nn.Parameter(
            torch.Tensor(num_queries, embed_dim))
        nn.init.xavier_uniform_(self.coattention_queries)

        self.seq_to_query_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.query_to_seq_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_emb, src_key_padding_mask=None):
        """
        Args:
            seq_emb: Tensor, shape (batch_size, seq_len, embed_dim) - embedded input sequence.
            src_key_padding_mask: Tensor, shape (batch_size, seq_len) - mask for padded elements in seq_emb.
        Returns:
            query_aware_seq: Tensor, shape (batch_size, seq_len, embed_dim) - sequence representation refined by co-attention.
        """
        batch_size = seq_emb.size(0)

        expanded_queries = self.coattention_queries.unsqueeze(
            0).repeat(batch_size, 1, 1)

        # Queries attend to sequence
        seq_aware_queries, _ = self.seq_to_query_attn(
            query=expanded_queries,
            key=seq_emb,
            value=seq_emb,
            key_padding_mask=src_key_padding_mask
        )

        # Sequence attends to sequence-aware queries
        query_aware_seq_intermediate, _ = self.query_to_seq_attn(
            query=seq_emb,
            key=seq_aware_queries,
            value=seq_aware_queries
        )

        query_aware_seq = self.norm1(
            seq_emb + self.dropout(query_aware_seq_intermediate))

        return query_aware_seq


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
