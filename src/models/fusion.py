import torch
import torch.nn as nn
from .vit_backbone import ViTExtractor
from .temporal import TemporalTransformer
from .frequency import FrequencyCNN
from .audio import AudioCNN

class MultiModalFusionModel(nn.Module):
    def __init__(self, vit_pretrained=True, dropout=0.4):
        super().__init__()
        
        # Sub-modules for spatial, temporal, frequency, and audio streams
        self.vit = ViTExtractor(use_pretrained=vit_pretrained)
        self.temporal = TemporalTransformer(embed_dim=self.vit.embed_dim)
        self.frequency = FrequencyCNN(embed_dim=256)
        self.audio = AudioCNN(embed_dim=256)
        
        # Fusion dimensionality: ViT Temporal (768) + FFT CNN (256) + Audio CNN (256) = 1280
        fusion_dim = self.vit.embed_dim + self.frequency.embed_dim + self.audio.embed_dim
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 1) # Single output for binary classification logit (BCEWithLogitsLoss)
        )

    def forward(self, rgb_seq, fft_seq, audio_spec):
        """
        Args:
            rgb_seq: (Batch, Sequence, 3, 224, 224)
            fft_seq: (Batch, Sequence, 1, 224, 224)
            audio_spec: (Batch, 128, time_steps)
        Returns:
            Tensor of shape (Batch, 1) representing fake probability logits.
        """
        B, S, C, H, W = rgb_seq.shape
        
        # 1. Process Spatial Path (ViT)
        rgb_flat = rgb_seq.view(B * S, C, H, W)
        vit_emb_flat = self.vit(rgb_flat)
        vit_emb = vit_emb_flat.view(B, S, -1) # Restore sequence dimension
        
        # 2. Process Temporal Path
        video_emb = self.temporal(vit_emb)
        
        # 3. Process Frequency Path (FFT)
        fft_emb = self.frequency(fft_seq)
        
        # 4. Process Audio Path
        audio_emb = self.audio(audio_spec)
        
        # 5. Connect and Classify
        fused = torch.cat([video_emb, fft_emb, audio_emb], dim=1) # (Batch, fusion_dim)
        logits = self.classifier(fused)
        
        return logits

if __name__ == "__main__":
    # Smoke test for tensor shape alignment
    model = MultiModalFusionModel()
    dummy_rgb = torch.randn(2, 16, 3, 224, 224)
    dummy_fft = torch.randn(2, 16, 1, 224, 224)
    dummy_aud = torch.randn(2, 128, 64)
    
    out = model(dummy_rgb, dummy_fft, dummy_aud)
    print("Multi-modal pipeline forward pass successful. Output shape:", out.shape)
