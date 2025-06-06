import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from model.modules import HallmarkFeatureProcessor, PositionalEncoding


class HallmarkSurvivalModel(nn.Module):
    """
    PyTorch model for multiclass survival classification using hallmark data.
    """

    def __init__(self,
                 M,
                 N_CLASSES,
                 hallmark_embedding_dim=256,  # Dimension of embedding from feature processing
                 cnn_filters=32,
                 cnn_kernel_size=3,
                 fc1_units=512,
                 fc2_units=256,
                 fc3_units=128,
                 dropout_rate=0.4,
                 num_transformer_heads=8,
                 num_transformer_layers=2):
        """
        Main network for multiclass survival classification with variable length hallmarks.

        Args:
            M (int): Number of hallmark gene sets.
            N_CLASSES (int): Number of survival classes (output bins).
            hallmark_embedding_dim (int): Fixed output dimension from the SharedHallmarkProcessor.
            cnn_filters (int): Num filters for the shared CNN processor.
            cnn_kernel_size (int): Kernel size for the shared CNN processor.
            fc1_units (int): Units in the first dense layer of the backbone.
            fc2_units (int): Units in the second dense layer of the backbone.
            dropout_rate (float): Dropout rate.
        """
        super(HallmarkSurvivalModel, self).__init__()
        self.M = M
        self.N_CLASSES = N_CLASSES

        # Instantiate the shared processor
        self.hallmark_processor = HallmarkFeatureProcessor(
            output_embedding_dim=hallmark_embedding_dim,
            cnn_filters=cnn_filters,
            cnn_kernel_size=cnn_kernel_size
        )

        self.positional_encoding = PositionalEncoding(
            d_model=hallmark_embedding_dim,
            max_len=M
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hallmark_embedding_dim,
            nhead=num_transformer_heads,
            dim_feedforward=4 * hallmark_embedding_dim,  # Typical feedforward dim
            dropout=dropout_rate,
            # Important: Input (batch_size, seq_len, features)
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        self.conv1 = nn.Conv2d(1, 3, 5)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Calculate the total number of features after processing all M hallmarks
        self.concatenated_features_dim = M * hallmark_embedding_dim
        self.hallmark_embedding_dim = hallmark_embedding_dim

        # Fully connected backbone
        self.fc1 = nn.Linear(self.concatenated_features_dim, fc1_units)
        self.bn_fc1 = nn.BatchNorm1d(fc1_units)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn_fc2 = nn.BatchNorm1d(fc2_units)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.bn_fc3 = nn.BatchNorm1d(fc3_units)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Output layer
        self.output_fc = nn.Linear(fc3_units, N_CLASSES)

    def forward_without_classification(self, list_of_hallmark_data):
        """
        Forward pass.

        Args:
            list_of_hallmark_data (list): A list of M tensors. 
                                          Each tensor list_of_hallmark_data[i] should have
                                          shape (batch_size, L_i), where L_i is the length
                                          of the i-th hallmark's feature vector.

        Returns:
            torch.Tensor: Logits of shape (batch_size, N_CLASSES).
        """
        if len(list_of_hallmark_data) != self.M:
            raise ValueError(
                f"Expected a list of {self.M} tensors, got {len(list_of_hallmark_data)}")

        processed_embeddings = []
        for i in range(self.M):
            x_i = list_of_hallmark_data[i]

            # Add channel dimension for Conv1D: (batch_size, 1, L_i)
            x_i_cnn_input = x_i.unsqueeze(1)

            # (batch_size, hallmark_embedding_dim)
            embedding_i = self.hallmark_processor(x_i_cnn_input)
            processed_embeddings.append(embedding_i)

        processed_embeddings = torch.stack(processed_embeddings, dim=1)
        embed_with_pos = self.positional_encoding(processed_embeddings)
        transformer_out = self.transformer_encoder(embed_with_pos)
        flattened = torch.flatten(transformer_out, start_dim=1)

        # Concatenate embeddings from all M hallmarks
        # List of M tensors of shape (batch_size, hallmark_embedding_dim)
        # -> single tensor of shape (batch_size, M * hallmark_embedding_dim)
        # concatenated_features = torch.cat(processed_embeddings, dim=1)

        out = self.fc1(flattened)
        out = self.bn_fc1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn_fc2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.bn_fc3(out)
        out = F.relu(out)
        out = self.dropout3(out)

        return out

    def forward(self, list_of_hallmark_data):
        out = self.forward_without_classification(list_of_hallmark_data)
        logits = self.output_fc(out)
        return logits


class AttentionFusion(nn.Module):
    """Attention-based fusion module"""
    def __init__(self, hallmark_dim, clinical_dim, hidden_dim=128):
        super(AttentionFusion, self).__init__()
        self.hallmark_proj = nn.Linear(hallmark_dim, hidden_dim)
        self.clinical_proj = nn.Linear(clinical_dim, hidden_dim)
        
        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=4, 
            batch_first=True
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, hallmark_features, clinical_features):
        # Project to same dimension
        h_proj = self.hallmark_proj(hallmark_features).unsqueeze(1)  # [B, 1, hidden_dim]
        c_proj = self.clinical_proj(clinical_features).unsqueeze(1)   # [B, 1, hidden_dim]
        
        # Concatenate into sequence
        combined = torch.cat([h_proj, c_proj], dim=1)  # [B, 2, hidden_dim]
        
        # Self-attention
        attn_out, _ = self.attention(combined, combined, combined)
        attn_out = self.norm1(attn_out + combined)
        
        # Cross-modal attention (hallmark attend to clinical)
        cross_out, _ = self.cross_attention(h_proj, c_proj, c_proj)
        cross_out = self.norm2(cross_out + h_proj)
        
        # Fusion output
        fused = torch.cat([attn_out.flatten(1), cross_out.flatten(1)], dim=1)
        return fused


class GatedFusion(nn.Module):
    """Gated fusion module"""
    def __init__(self, hallmark_dim, clinical_dim, output_dim=128):
        super(GatedFusion, self).__init__()
        self.hallmark_transform = nn.Linear(hallmark_dim, output_dim)
        self.clinical_transform = nn.Linear(clinical_dim, output_dim)
        
        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(hallmark_dim + clinical_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 2),  # Weights for 2 modalities
            nn.Softmax(dim=1)
        )
        
        # Interaction learning
        self.interaction = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, hallmark_features, clinical_features):
        # Transform to same dimension
        h_transformed = self.hallmark_transform(hallmark_features)
        c_transformed = self.clinical_transform(clinical_features)
        
        # Compute gating weights
        concat_features = torch.cat([hallmark_features, clinical_features], dim=1)
        gates = self.gate_network(concat_features)
        
        # Weighted fusion
        weighted_h = h_transformed * gates[:, 0:1]
        weighted_c = c_transformed * gates[:, 1:2]
        
        # Interaction learning
        interaction_input = torch.cat([weighted_h, weighted_c], dim=1)
        fused = self.interaction(interaction_input)
        
        return fused


class MultimodalHallmarkSurvivalModel(nn.Module):

    def __init__(self,
                 M,
                 N_CLASSES,
                 clinical_input_dim,
                 hallmark_embedding_dim=256,
                 cnn_filters=32,
                 cnn_kernel_size=3,
                 fc1_units=512,
                 fc2_units=256,
                 fc3_units=128,
                 clinical_fc_units=64,
                 dropout_rate=0.4,
                 num_transformer_heads=8,
                 num_transformer_layers=2,
                 fusion_method='attention'):  # 'concat', 'attention', 'gated'
        super(MultimodalHallmarkSurvivalModel, self).__init__()
        self.fusion_method = fusion_method
        
        self.hallmark_model = HallmarkSurvivalModel(
            M=M,
            N_CLASSES=N_CLASSES,
            hallmark_embedding_dim=hallmark_embedding_dim,
            cnn_filters=cnn_filters,
            cnn_kernel_size=cnn_kernel_size,
            fc1_units=fc1_units,
            fc2_units=fc2_units,
            fc3_units=fc3_units,
            dropout_rate=dropout_rate,
            num_transformer_heads=num_transformer_heads,
            num_transformer_layers=num_transformer_layers
        )
        
        # Improved clinical feature processing
        self.clinical_fc = nn.Sequential(
            nn.Linear(clinical_input_dim, clinical_fc_units * 2),
            nn.BatchNorm1d(clinical_fc_units * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(clinical_fc_units * 2, clinical_fc_units),
            nn.BatchNorm1d(clinical_fc_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Select fusion strategy
        if fusion_method == 'attention':
            self.fusion_layer = AttentionFusion(fc3_units, clinical_fc_units)
            fusion_output_dim = 256 + 128  # attention fusion output
        elif fusion_method == 'gated':
            self.fusion_layer = GatedFusion(fc3_units, clinical_fc_units, output_dim=128)
            fusion_output_dim = 128
        else:  # concat
            self.fusion_layer = None
            fusion_output_dim = fc3_units + clinical_fc_units
        
        # Improved classification network
        self.combined_fc = nn.Sequential(
            nn.Linear(fusion_output_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, N_CLASSES)
        )
        
        # Add modality-specific regularization
        self.hallmark_l2_reg = 0.001
        self.clinical_l2_reg = 0.001

    def forward(self, list_of_hallmark_data, clinical_data):
        # Hallmark branch
        hallmark_features = self.hallmark_model.forward_without_classification(
            list_of_hallmark_data
        )
        
        # Clinical branch
        clinical_features = self.clinical_fc(clinical_data)
        
        # Feature fusion
        if self.fusion_method == 'attention':
            combined = self.fusion_layer(hallmark_features, clinical_features)
        elif self.fusion_method == 'gated':
            combined = self.fusion_layer(hallmark_features, clinical_features)
        else:  # concat
            combined = torch.cat([hallmark_features, clinical_features], dim=1)
        
        # Final classification
        logits = self.combined_fc(combined)
        return logits
    
    def get_regularization_loss(self):
        """Get regularization loss"""
        hallmark_reg = 0
        for param in self.hallmark_model.parameters():
            hallmark_reg += torch.norm(param, 2)
        
        clinical_reg = 0
        for param in self.clinical_fc.parameters():
            clinical_reg += torch.norm(param, 2)
            
        return self.hallmark_l2_reg * hallmark_reg + self.clinical_l2_reg * clinical_reg
