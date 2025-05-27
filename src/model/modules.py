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
