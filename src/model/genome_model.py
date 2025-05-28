import torch.nn as nn
from model.modules import PositionalEncoding, CoAttentionModule
import numpy as np


class GenomePrediction(nn.Module):
    """
    CoAttention Transformer model for RNA gene expression classification.
    """

    def __init__(self, num_classes, num_genes, embed_dim, num_coattention_queries,
                 num_attn_heads, num_transformer_layers, dim_feedforward, dropout=0.1):
        super(GenomePrediction, self).__init__()

        self.embed_dim = embed_dim
        self.num_genes = num_genes  # Number of gene features, also acts as sequence length

        # 1. Input Projection Layer for Gene Expression Values
        # Each gene's expression value (a scalar) is projected to embed_dim.
        # Input to this layer will be (batch_size, num_genes, 1)
        # Output will be (batch_size, num_genes, embed_dim)
        self.input_projection = nn.Linear(1, embed_dim)

        # 2. Positional Encoding
        # max_len for positional encoding is num_genes.
        self.pos_encoder = PositionalEncoding(
            embed_dim, dropout, max_len=num_genes)

        # 3. CoAttention Module
        self.coattention_module = CoAttentionModule(
            embed_dim, num_attn_heads, num_coattention_queries, dropout)

        # 4. Standard Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_attn_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers)

        # 5. Classifier Head
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.final_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        # Initialize weights for linear layers and coattention queries.
        for p in self.parameters():
            if p.dim() > 1:  # For weight matrices
                nn.init.xavier_uniform_(p)
        # Specific initialization for input_projection if desired, e.g.:
        # nn.init.kaiming_normal_(self.input_projection.weight, mode='fan_in', nonlinearity='relu')
        # if self.input_projection.bias is not None:
        #     nn.init.constant_(self.input_projection.bias, 0)

    def forward(self, src_gene_expressions, src_key_padding_mask=None):
        """
        Args:
            src_gene_expressions: Tensor, shape (batch_size, num_genes) 
                                  - float values of gene expressions.
            src_key_padding_mask: Tensor, shape (batch_size, num_genes) 
                                  - boolean mask, True for genes to be masked/padded.
        Returns:
            logits: Tensor, shape (batch_size, num_classes) 
                    - raw output scores for each class.
        """

        # 1. Input Projection
        # Reshape src_gene_expressions from (batch_size, num_genes) to (batch_size, num_genes, 1)
        x = src_gene_expressions.unsqueeze(2)

        # Project to (batch_size, num_genes, embed_dim)
        x = self.input_projection(x)
        # Optional: Scale projected embeddings, similar to how token embeddings are often scaled.
        x = x * np.sqrt(self.embed_dim)

        # 2. Positional Encoding
        x = self.pos_encoder(x)  # Output: (batch_size, num_genes, embed_dim)

        # 3. CoAttention Module
        # Output: (batch_size, num_genes, embed_dim)
        x = self.coattention_module(
            x, src_key_padding_mask=src_key_padding_mask)

        # 4. Transformer Encoder
        # src_key_padding_mask masks attention over specified genes.
        # Output: (batch_size, num_genes, embed_dim)
        x = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask)

        # 5. Classification - Pooling and Linear Layer
        # Mean pooling over the gene dimension, handling padding if mask is provided.
        if src_key_padding_mask is not None:
            # (batch_size, num_genes)
            active_elements_mask = ~src_key_padding_mask
            # Zero out masked genes
            x_masked = x * active_elements_mask.unsqueeze(-1)
            num_active_elements = active_elements_mask.sum(
                dim=1, keepdim=True).clamp(min=1.0)
            pooled_output = x_masked.sum(dim=1) / num_active_elements
        else:
            pooled_output = x.mean(dim=1)  # (batch_size, embed_dim)

        pooled_output = self.final_dropout(pooled_output)
        logits = self.classifier(pooled_output)  # (batch_size, num_classes)

        return logits
