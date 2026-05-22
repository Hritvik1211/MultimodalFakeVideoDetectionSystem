import torch
import torch.nn as nn
import torchvision.models as models

class ViTExtractor(nn.Module):
    def __init__(self, use_pretrained=True):
        super().__init__()
        weights = models.ViT_B_16_Weights.DEFAULT if use_pretrained else None
        self.vit = models.vit_b_16(weights=weights)
        
        # Remove the final classification head to output raw embeddings
        self.vit.heads = nn.Identity()
        
        # ViT-B/16 output embed dimension
        self.embed_dim = 768

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, 3, 224, 224)
        Returns:
            Tensor of shape (Batch, 768)
        """
        return self.vit(x)

if __name__ == "__main__":
    model = ViTExtractor()
    dummy_in = torch.randn(2, 3, 224, 224)
    out = model(dummy_in)
    print("ViT output shape:", out.shape)
