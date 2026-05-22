import torch
import torch.nn as nn

class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=4, dropout=0.3):
        super().__init__()
        # Learnable CLASS token to aggregate temporal sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Sequence, Embed_Dim)
        Returns:
            Tensor of shape (Batch, Embed_Dim) representing aggregated video embedding.
        """
        B, S, E = x.shape
        
        # Expand cls_token across batch
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, E)
        
        # Prepend to sequence
        x = torch.cat((cls_tokens, x), dim=1) # (B, S+1, E)
        
        # Pass through temporal transformer logic
        out = self.transformer(x)
        
        # Extract the cls_token output representations
        video_emb = out[:, 0, :] # (B, E)
        return video_emb
