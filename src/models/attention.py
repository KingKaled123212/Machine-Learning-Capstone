"""
Attention-based Architecture (Bonus)
Simplified Vision Transformer for images
"""
import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them"""
    
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Linear projections
        Q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.out_linear(context)
        return output


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, d_model, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x))
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Simplified Vision Transformer for image classification
    
    Features:
    - Patch-based image embedding
    - Multi-head self-attention
    - Transformer encoder blocks
    - Classification head
    """
    
    def __init__(self, input_dim, num_classes, d_model=128, num_heads=4, 
                 num_layers=2, patch_size=4, dropout=0.1, fc_dims=[128, 64]):
        """
        Args:
            input_dim: Input dimension (C, H, W)
            num_classes: Number of output classes
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            patch_size: Size of image patches
            dropout: Dropout probability
            fc_dims: Classifier head dimensions
        """
        super().__init__()
        
        # Handle input dimensions
        if isinstance(input_dim, tuple):
            in_channels, img_size = input_dim[0], input_dim[1]
        else:
            in_channels, img_size = 1, input_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        n_patches = self.patch_embed.n_patches
        
        # Learnable position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, d_model))
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Dropout
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Classification head
        head_layers = []
        prev_dim = d_model
        
        for fc_dim in fc_dims:
            head_layers.extend([
                nn.Linear(prev_dim, fc_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = fc_dim
        
        head_layers.append(nn.Linear(prev_dim, num_classes))
        self.head = nn.Sequential(*head_layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B = x.size(0)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, d_model)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, d_model)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Classification from class token
        cls_output = x[:, 0]
        output = self.head(cls_output)
        
        return output
    
    def get_param_count(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionMLP(nn.Module):
    """
    Attention-based MLP for tabular data
    Simple self-attention mechanism for feature importance
    """
    
    def __init__(self, input_dim, num_classes, d_model=128, num_heads=4, 
                 dropout=0.1, fc_dims=[128, 64]):
        super().__init__()
        
        if isinstance(input_dim, tuple):
            input_dim = int(torch.prod(torch.tensor(input_dim)))
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Self-attention
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        
        # MLP head
        head_layers = []
        prev_dim = d_model
        
        for fc_dim in fc_dims:
            head_layers.extend([
                nn.Linear(prev_dim, fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = fc_dim
        
        head_layers.append(nn.Linear(prev_dim, num_classes))
        self.head = nn.Sequential(*head_layers)
    
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Project to d_model
        x = self.input_proj(x).unsqueeze(1)  # (B, 1, d_model)
        
        # Self-attention
        x = self.attn(x)
        x = self.norm(x)
        
        # Classification
        x = x.squeeze(1)
        output = self.head(x)
        
        return output
    
    def get_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test Vision Transformer
    print("Testing Vision Transformer...")
    model = VisionTransformer(input_dim=(3, 32, 32), num_classes=10, patch_size=4)
    x = torch.randn(16, 3, 32, 32)
    output = model(x)
    print(f"Input shape: {x.shape}, output shape: {output.shape}")
    print(f"Parameters: {model.get_param_count():,}")
    
    # Test Attention MLP
    print("\nTesting Attention MLP...")
    model = AttentionMLP(input_dim=14, num_classes=2)
    x = torch.randn(32, 14)
    output = model(x)
    print(f"Input shape: {x.shape}, output shape: {output.shape}")
    print(f"Parameters: {model.get_param_count():,}")
