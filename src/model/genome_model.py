import torch.nn as nn
import torch.nn.functional as F
import torch

from model.modules import HallmarkFeatureProcessor, PositionalEncoding


class HallmarkSurvivalModel(nn.Module):
    """
    PyTorch model for multiclass survival classification using hallmark data.
    """

    def __init__(self,
                 M,
                 N_CLASSES,
                 hallmark_embedding_dim=64,  # Dimension of embedding from feature processing
                 cnn_filters=8,
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

    def forward(self, list_of_hallmark_data):
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

        logits = self.output_fc(out)

        return logits
