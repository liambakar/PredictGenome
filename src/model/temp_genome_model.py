import torch.nn as nn


class Simple(nn.Module):
    def __init__(self, num_genes, num_classes, hidden_dim1=128, hidden_dim2=64, dropout_rate=0.3):
        """
        A simple Multi-Layer Perceptron (MLP) for gene expression classification.

        Args:
            num_genes (int): The number of input gene features.
            num_classes (int): The number of output classes.
            hidden_dim1 (int): The number of neurons in the first hidden layer.
            hidden_dim2 (int): The number of neurons in the second hidden layer.
            dropout_rate (float): Dropout probability for regularization.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_genes, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Apply dropout after activation
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Apply dropout after activation
            nn.Linear(hidden_dim2, num_classes),
            nn.LogSoftmax(dim=1)
        )
        self._init_weights()

    def _init_weights(self):
        # Initialize weights for linear layers
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_genes)
                              containing gene expression values.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
                          containing raw logits for each class.
        """
        return self.network(x)
