import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # Input is 1 channel Log Mel Spectrogram (128 mel bins, time_steps)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, 128, time_steps)
        Returns:
            Tensor of shape (Batch, embed_dim)
        """
        # Add channel dimension to mimic an image input
        x = x.unsqueeze(1) # (Batch, 1, 128, time_steps)
        
        features = self.net(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)
