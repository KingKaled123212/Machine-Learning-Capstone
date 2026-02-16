"""
Convolutional Neural Network (CNN) Architecture
"""
import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Convolutional Neural Network for image classification
    
    Features:
    - Configurable conv layers
    - Batch normalization
    - Max pooling
    - Dropout regularization
    - Fully connected classifier head
    """
    
    def __init__(self, input_dim, num_classes, conv_channels=[32, 64, 128],
                 kernel_sizes=[3, 3, 3], pool_sizes=[2, 2, 2], 
                 fc_dims=[256, 128], dropout=0.3, use_batch_norm=True):
        """
        Args:
            input_dim: Input dimension (C, H, W)
            num_classes: Number of output classes
            conv_channels: List of output channels for conv layers
            kernel_sizes: List of kernel sizes for conv layers
            pool_sizes: List of pool sizes
            fc_dims: List of fully connected layer dimensions
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(CNN, self).__init__()
        
        # Handle both tuple and int input
        if isinstance(input_dim, int):
            input_channels = 1
            self.input_size = (input_dim, input_dim)
        else:
            input_channels = input_dim[0]
            self.input_size = input_dim[1:]
        
        # Convolutional layers
        conv_layers = []
        in_channels = input_channels
        
        for i, out_channels in enumerate(conv_channels):
            # Conv layer
            conv_layers.append(nn.Conv2d(
                in_channels, 
                out_channels,
                kernel_size=kernel_sizes[i],
                padding=kernel_sizes[i]//2
            ))
            
            # Batch normalization
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            
            # Activation
            conv_layers.append(nn.ReLU())
            
            # Pooling
            if i < len(pool_sizes):
                conv_layers.append(nn.MaxPool2d(pool_sizes[i]))
            
            # Dropout
            if dropout > 0:
                conv_layers.append(nn.Dropout2d(dropout * 0.5))  # Lower dropout for conv
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate flattened size
        self.flatten_size = self._get_flatten_size(input_dim)
        
        # Fully connected layers
        fc_layers = []
        prev_dim = self.flatten_size
        
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(prev_dim, fc_dim))
            
            if use_batch_norm:
                fc_layers.append(nn.BatchNorm1d(fc_dim))
            
            fc_layers.append(nn.ReLU())
            
            if dropout > 0:
                fc_layers.append(nn.Dropout(dropout))
            
            prev_dim = fc_dim
        
        # Output layer
        fc_layers.append(nn.Linear(prev_dim, num_classes))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_flatten_size(self, input_dim):
        """Calculate the size after conv layers"""
        if isinstance(input_dim, int):
            x = torch.randn(1, 1, input_dim, input_dim)
        else:
            x = torch.randn(1, *input_dim)
        
        x = self.conv_layers(x)
        return x.view(1, -1).size(1)
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        # Handle 1D input (for tabular data reshaped to 2D)
        if x.dim() == 2:
            # Reshape to square image
            size = int(x.size(1) ** 0.5)
            if size * size < x.size(1):
                size += 1
            # Pad if necessary
            if size * size > x.size(1):
                padding = size * size - x.size(1)
                x = torch.cat([x, torch.zeros(x.size(0), padding, device=x.device)], dim=1)
            x = x.view(-1, 1, size, size)
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
    def get_param_count(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test CNN
    print("Testing CNN Architecture...")
    
    # Test with CIFAR-10 style images
    model = CNN(input_dim=(3, 32, 32), num_classes=10)
    x = torch.randn(32, 3, 32, 32)
    output = model(x)
    print(f"Image input shape: {x.shape}, output shape: {output.shape}")
    print(f"Parameters: {model.get_param_count():,}")
    
    # Test with larger images (PCam style)
    model = CNN(input_dim=(3, 96, 96), num_classes=2)
    x = torch.randn(16, 3, 96, 96)
    output = model(x)
    print(f"\nLarge image input shape: {x.shape}, output shape: {output.shape}")
    print(f"Parameters: {model.get_param_count():,}")
