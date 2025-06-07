import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    Simple GCN layer implementation
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        Forward pass for GCN layer
        Args:
            x: Node features (batch_size, num_nodes, in_features)
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes)
        """
        # x shape: (batch_size, num_nodes, in_features)
        # adj shape: (batch_size, num_nodes, num_nodes)

        # Apply weight transformation: (batch_size, num_nodes, out_features)
        support = torch.matmul(x, self.weight)

        # Apply adjacency matrix: (batch_size, num_nodes, out_features)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            output = output + self.bias

        return output


class FeatureProcessor(nn.Module):
    def __init__(self, embedding_dim=64, cnn_filters=32, cnn_kernel_size=3):
        super(FeatureProcessor, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel_size,
            padding=(cnn_kernel_size - 1) // 2
        )
        self.bn1 = nn.BatchNorm1d(cnn_filters)

        # Adaptive pooling to handle variable lengths
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)

        # Project to embedding dimension
        self.fc = nn.Linear(cnn_filters, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        """
        Args:
            x: Variable length feature (batch_size, length)
        Returns:
            embedding: Fixed size embedding (batch_size, embedding_dim)
        """
        # Add channel dimension: (batch_size, 1, length)
        x = x.unsqueeze(1)

        # Apply convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Adaptive pooling: (batch_size, cnn_filters, 1)
        x = self.adaptive_pool(x)
        x = x.squeeze(2)  # (batch_size, cnn_filters)

        # Project to embedding
        x = self.fc(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x


class GCNGenomicModel(nn.Module):
    """
    GCN model for genomic data with 50 variable-length features
    """

    def __init__(self,
                 num_features=50,
                 num_classes=4,
                 feature_embedding_dim=64,
                 gcn_hidden_dims=[128, 64],
                 classifier_dims=[256, 128, 64],
                 dropout_rate=0.4):
        super(GCNGenomicModel, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.feature_embedding_dim = feature_embedding_dim

        # Feature processor for each variable-length feature
        self.feature_processor = FeatureProcessor(
            embedding_dim=feature_embedding_dim)

        # Build GCN layers
        gcn_layers = []
        in_dim = feature_embedding_dim
        for hidden_dim in gcn_hidden_dims:
            gcn_layers.append(GCNLayer(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.gcn_layers = nn.ModuleList(gcn_layers)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Global pooling for graph-level representation
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # Classifier
        classifier_layers = []
        in_dim = gcn_hidden_dims[-1] if gcn_hidden_dims else feature_embedding_dim

        for hidden_dim in classifier_dims:
            classifier_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_dim = hidden_dim

        classifier_layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

        # Create adjacency matrix (learnable or predefined)
        self.create_adjacency_matrix()

    def create_adjacency_matrix(self):
        """
        Create adjacency matrix for the feature graph
        Here we use a learnable adjacency matrix, but you can modify this
        to use domain knowledge or correlation-based adjacency
        """
        # Learnable adjacency matrix
        self.adj_matrix = nn.Parameter(
            torch.randn(self.num_features, self.num_features)
        )

    def get_adjacency_matrix(self, batch_size):
        """
        Get normalized adjacency matrix for the batch
        """
        # Apply softmax to make it a proper adjacency matrix
        adj = F.softmax(self.adj_matrix, dim=1)

        # Add self-connections
        adj = adj + torch.eye(self.num_features, device=adj.device)

        # Normalize (simple row normalization)
        degree = torch.sum(adj, dim=1, keepdim=True)
        adj_normalized = adj / (degree + 1e-8)

        # Expand for batch
        adj_batch = adj_normalized.unsqueeze(0).expand(batch_size, -1, -1)

        return adj_batch

    def forward(self, feature_list):
        """
        Forward pass
        Args:
            feature_list: List of 50 tensors, each with shape (batch_size, variable_length)
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = feature_list[0].shape[0]

        # Process each feature to get fixed-size embeddings
        feature_embeddings = []
        for feature in feature_list:
            embedding = self.feature_processor(
                feature)  # (batch_size, embedding_dim)
            feature_embeddings.append(embedding)

        # Stack to create node feature matrix
        # (batch_size, num_features, embedding_dim)
        x = torch.stack(feature_embeddings, dim=1)

        # Get adjacency matrix
        adj = self.get_adjacency_matrix(batch_size)

        # Apply GCN layers
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, adj)
            x = F.relu(x)
            x = self.dropout(x)

        # Global pooling to get graph-level representation
        # x shape: (batch_size, num_features, hidden_dim)
        # Transpose and pool: (batch_size, hidden_dim, num_features) -> (batch_size, hidden_dim, 1)
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, num_features)
        x = self.global_pool(x)  # (batch_size, hidden_dim, 1)
        x = x.squeeze(2)  # (batch_size, hidden_dim)

        # Classification
        logits = self.classifier(x)

        return logits


class CorrelationBasedGCNModel(GCNGenomicModel):
    """
    GCN model that uses correlation-based adjacency matrix
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove learnable adjacency matrix
        delattr(self, 'adj_matrix')

    def get_adjacency_matrix(self, batch_size, feature_embeddings=None):
        """
        Create adjacency matrix based on feature correlation
        """
        if feature_embeddings is None:
            # Use identity matrix if no embeddings provided
            adj = torch.eye(self.num_features)
            adj_batch = adj.unsqueeze(0).expand(batch_size, -1, -1)
            return adj_batch.to(next(self.parameters()).device)

        # Compute correlation matrix
        # feature_embeddings shape: (batch_size, num_features, embedding_dim)
        batch_adj = []

        for b in range(batch_size):
            embeddings = feature_embeddings[b]  # (num_features, embedding_dim)

            # Compute correlation matrix
            embeddings_norm = F.normalize(embeddings, dim=1)
            corr_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)

            # Apply threshold to make it sparse (optional)
            threshold = 0.1
            corr_matrix = torch.where(
                torch.abs(corr_matrix) > threshold,
                corr_matrix,
                torch.zeros_like(corr_matrix)
            )

            # Add self-connections
            corr_matrix = corr_matrix + \
                torch.eye(self.num_features, device=corr_matrix.device)

            # Normalize
            degree = torch.sum(torch.abs(corr_matrix), dim=1, keepdim=True)
            corr_matrix = corr_matrix / (degree + 1e-8)

            batch_adj.append(corr_matrix)

        return torch.stack(batch_adj, dim=0)

    def forward(self, feature_list):
        """
        Forward pass with correlation-based adjacency
        """
        batch_size = feature_list[0].shape[0]

        # Process each feature to get fixed-size embeddings
        feature_embeddings = []
        for feature in feature_list:
            embedding = self.feature_processor(feature)
            feature_embeddings.append(embedding)

        # Stack to create node feature matrix
        x = torch.stack(feature_embeddings, dim=1)

        # For first GCN layer, use correlation-based adjacency
        if len(self.gcn_layers) > 0:
            adj = self.get_adjacency_matrix(batch_size, x)
            x = self.gcn_layers[0](x, adj)
            x = F.relu(x)
            x = self.dropout(x)

            # For subsequent layers, use simple adjacency
            for gcn_layer in self.gcn_layers[1:]:
                adj = self.get_adjacency_matrix(batch_size)  # Simple adjacency
                x = gcn_layer(x, adj)
                x = F.relu(x)
                x = self.dropout(x)

        # Global pooling and classification
        x = x.transpose(1, 2)
        x = self.global_pool(x)
        x = x.squeeze(2)

        logits = self.classifier(x)
        return logits
