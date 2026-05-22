import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path

class FakeVideoDataset(Dataset):
    def __init__(self, metadata_csv, num_frames=16, transform=None):
        """
        Args:
            metadata_csv (str): Path to metadata file containing paths to parsed RGB, FFT, Audio data.
            num_frames (int): Number of frames saved per video sequence.
            transform (callable, optional): PyTorch transforms for the RGB sequence.
        """
        self.metadata = pd.read_csv(metadata_csv)
        # Keep successful AND partial extractions (partial = RGB+FFT ok, audio missing)
        if 'status' in self.metadata.columns:
            self.metadata = self.metadata[
                self.metadata['status'].isin(['success', 'partial'])
            ]
        self.num_frames = num_frames
        
        # Standard ViT transform strategy matching ViT-B/16
        self.transform = transform if transform else T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # FFT specific transform (single channel, grayscale)
        self.fft_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        rgb_dir = Path(row['rgb_path'])
        fft_dir = Path(row['fft_path'])
        audio_path = Path(row['audio_path'])
        
        # Assuming label is 0=Real, 1=Fake for BCE loss
        label = torch.tensor(row['label'], dtype=torch.float32)
        
        rgb_frames = []
        fft_frames = []
        
        for i in range(self.num_frames):
            rgb_f = rgb_dir / f"frame_{i:03d}.jpg"
            fft_f = fft_dir / f"frame_{i:03d}.jpg"
            
            # --- RGB ---
            if rgb_f.exists():
                rgb_img = Image.open(rgb_f).convert('RGB')
                rgb_frames.append(self.transform(rgb_img))
            else:
                # Black padding if missing
                rgb_frames.append(torch.zeros(3, 224, 224))
                
            # --- FFT ---
            if fft_f.exists():
                fft_img = Image.open(fft_f).convert('L')
                fft_frames.append(self.fft_transform(fft_img))
            else:
                fft_frames.append(torch.zeros(1, 224, 224))
                
        # output sequence shape for frames: (num_frames, C, H, W)
        rgb_tensor = torch.stack(rgb_frames)
        fft_tensor = torch.stack(fft_frames)
        
        # --- AUDIO ---
        if audio_path.exists():
            audio_np = np.load(audio_path) # e.g., (128 mel bins, time_steps)
            audio_tensor = torch.from_numpy(audio_np).float()
            
            # Unify audio length (e.g., center crop to 64 steps, or pad)
            target_length = 64
            if audio_tensor.shape[1] > target_length:
                # Truncate
                audio_tensor = audio_tensor[:, :target_length]
            elif audio_tensor.shape[1] < target_length:
                # Pad
                pad_size = target_length - audio_tensor.shape[1]
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_size))
        else:
            audio_tensor = torch.zeros((128, 64))
            
        return {
            'rgb': rgb_tensor,
            'fft': fft_tensor,
            'audio': audio_tensor,
            'label': label,
            'video_id': row['video_id']
        }

if __name__ == "__main__":
    # Test dataset instantiation
    dummy_csv = "dataset/metadata.csv"
    if os.path.exists(dummy_csv):
        print("Testing FakeVideoDataset...")
        dataset = FakeVideoDataset(dummy_csv)
        print(f"Dataset length: {len(dataset)}")
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"RGB Shape: {sample['rgb'].shape}")
            print(f"FFT Shape: {sample['fft'].shape}")
            print(f"Audio Shape: {sample['audio'].shape}")
            print(f"Label: {sample['label']}")
