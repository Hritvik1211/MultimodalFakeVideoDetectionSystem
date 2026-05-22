import torch
import torch.nn as nn

class FrequencyCNN(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # Input is 1 channel FFT magnitude image (H x W)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, embed_dim)
        self.embed_dim = embed_dim
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Sequence, 1, H, W)
        Returns:
            Tensor of shape (Batch, embed_dim)
        """
        B, S, C, H, W = x.shape
        # Flatten batch and temporal dimension for convolutions
        x = x.view(B * S, C, H, W)
        
        features = self.net(x)
        features = features.view(B * S, -1)
        features = self.fc(features)
        
        # Reshape to separate Batch and Sequence dimensions
        features = features.view(B, S, -1)
        
        # Pool across sequence dimension (average FFT anomaly signature)
        return features.mean(dim=1)
