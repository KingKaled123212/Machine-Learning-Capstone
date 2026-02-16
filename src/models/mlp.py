"""
Multilayer Perceptron (MLP) Architecture
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multilayer Perceptron with configurable depth and width
    
    Features:
    - Configurable hidden layers
    - Batch normalization (optional)
    - Dropout regularization
    - ReLU activation
    """
    
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128, 64], 
                 dropout=0.3, use_batch_norm=True):
        """
        Args:
            input_dim: Input feature dimension (int or tuple)
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(MLP, self).__init__()
        
        # Handle tuple input_dim (for image data)
        if isinstance(input_dim, tuple):
            input_dim = int(torch.prod(torch.tensor(input_dim)))
        
        self.input_dim = input_dim
        self.flatten = nn.Flatten()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        x = self.flatten(x)
        return self.network(x)
    
    def get_param_count(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test MLP
    print("Testing MLP Architecture...")
    
    # Test with tabular data
    model = MLP(input_dim=14, num_classes=2)
    x = torch.randn(32, 14)
    output = model(x)
    print(f"Tabular input shape: {x.shape}, output shape: {output.shape}")
    print(f"Parameters: {model.get_param_count():,}")
    
    # Test with image data
    model = MLP(input_dim=(3, 32, 32), num_classes=10)
    x = torch.randn(32, 3, 32, 32)
    output = model(x)
    print(f"\nImage input shape: {x.shape}, output shape: {output.shape}")
    print(f"Parameters: {model.get_param_count():,}")
