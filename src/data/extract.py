import os
import cv2
import numpy as np
import pandas as pd
import librosa
import subprocess
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch

try:
    from facenet_pytorch import MTCNN
except ImportError:
    raise ImportError("Please install facenet_pytorch: pip install facenet-pytorch")

class VideoExtractor:
    def __init__(self, output_base, num_frames=16, target_size=(224, 224), use_gpu=True):
        self.output_base = Path(output_base)
        self.num_frames = num_frames
        self.target_size = target_size
        
        # Setup device for MTCNN
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print(f"Using device: {self.device}")
        
        self.mtcnn = MTCNN(keep_all=True, device=self.device, thresholds=[0.6, 0.7, 0.7], post_process=False)
        self.margin = 0.20
        self.confidence_threshold = 0.90

        # Create base metadata file if it doesn't exist
        self.metadata_path = self.output_base / "metadata.csv"
        if not self.metadata_path.exists():
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=["video_id", "label", "dataset", "rgb_path", "fft_path", "audio_path", "status"]).to_csv(self.metadata_path, index=False)

    def process_video(self, video_path, label, dataset_name):
        video_path = Path(video_path)
        video_id = video_path.stem
        
        # Setup paths
        vid_dir = self.output_base / f"{dataset_name}_{video_id}"
        rgb_dir = vid_dir / "rgb"
        fft_dir = vid_dir / "fft"
        audio_path = vid_dir / "audio.npy"
        
        rgb_dir.mkdir(parents=True, exist_ok=True)
        fft_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Extract Audio
        audio_success = self._extract_audio(video_path, audio_path)
        
        # 2. Extract and Process Frames
        frames_success = self._extract_frames(video_path, rgb_dir, fft_dir)
        
        status = "success" if (audio_success and frames_success) else "partial" if frames_success else "failed"
        
        # Log to metadata
        new_row = pd.DataFrame([{
            "video_id": f"{dataset_name}_{video_id}",
            "label": label,
            "dataset": dataset_name,
            "rgb_path": str(rgb_dir),
            "fft_path": str(fft_dir),
            "audio_path": str(audio_path),
            "status": status
        }])
        new_row.to_csv(self.metadata_path, mode='a', header=False, index=False)
        
        return status
        
    def _extract_audio(self, video_path, audio_out_path):
        """Extracts audio and saves log mel spectrogram"""
        temp_wav = audio_out_path.with_suffix(".wav")
        try:
            # Requires FFmpeg
            subprocess.run([
                "ffmpeg", "-i", str(video_path),
                "-q:a", "0", "-map", "a", str(temp_wav), "-y"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            # Load with librosa
            y, sr = librosa.load(temp_wav, sr=16000)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            np.save(audio_out_path, log_mel_spec)
            temp_wav.unlink()  # Clean up temp wav
            return True
        except Exception as e:
            # If no audio stream or ffmpeg fails
            if temp_wav.exists():
                temp_wav.unlink()
            
            # Create dummy audio to maintain pipeline structure (e.g. 1 sec silence)
            dummy_mel = np.zeros((128, 32))
            np.save(audio_out_path, dummy_mel)
            return False

    def _extract_frames(self, video_path, rgb_dir, fft_dir):
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return False

        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        success_count = 0
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                # If frame not found, we copy the previous one or use black box
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # MTCNN Face Detection
            boxes, probs = self.mtcnn.detect(pil_img)
            face_img = None
            
            if boxes is not None and len(boxes) > 0:
                valid = [(b, p) for b, p in zip(boxes, probs) if p is not None and p >= self.confidence_threshold]
                if valid:
                    boxes, probs = zip(*valid)
                    best_idx = np.argmax(probs)
                    box = boxes[best_idx]
                    
                    x1, y1, x2, y2 = box
                    fw, fh = x2 - x1, y2 - y1
                    mw, mh = fw * self.margin, fh * self.margin
                    
                    x1 = max(0, int(x1 - mw))
                    y1 = max(0, int(y1 - mh))
                    x2 = min(pil_img.width, int(x2 + mw))
                    y2 = min(pil_img.height, int(y2 + mh))
                    
                    face_img = pil_img.crop((x1, y1, x2, y2))
            
            # Fallback if no face found: Center crop
            if face_img is None:
                w, h = pil_img.size
                min_dim = min(w, h)
                left = (w - min_dim)/2
                top = (h - min_dim)/2
                face_img = pil_img.crop((left, top, left+min_dim, top+min_dim))
            
            # Resize
            face_img = face_img.resize(self.target_size, Image.BILINEAR)
            face_np = np.array(face_img)
            
            # Save RGB
            rgb_path = rgb_dir / f"frame_{i:03d}.jpg"
            cv2.imwrite(str(rgb_path), cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR))
            
            # 3. Frequency Analysis (FFT)
            # Convert to Grayscale for FFT
            gray_face = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)
            f = np.fft.fft2(gray_face)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
            
            # Normalize to 0-255 for saving as image
            mag_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            fft_path = fft_dir / f"frame_{i:03d}.jpg"
            cv2.imwrite(str(fft_path), mag_norm)
            
            success_count += 1
            
        cap.release()
        return success_count == self.num_frames

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, help="Directory containing videos or single video path")
    parser.add_argument("--output_dir", type=str, default="dataset/processed", help="Output directory mapping")
    parser.add_argument("--label", type=int, default=0, help="0 for Real, 1 for Fake")
    parser.add_argument("--dataset", type=str, default="dfdc", help="Dataset name e.g. dfdc, celeb-df")
    args = parser.parse_args()
    
    extractor = VideoExtractor(output_base=args.output_dir, num_frames=16, target_size=(224, 224))
    
    vid_path = Path(args.video_dir)
    
    # Parse available metadata*.json files to map video names to labels automatically
    labels_dict = {}
    if vid_path.is_dir():
        import json
        for json_path in vid_path.glob("metadata*.json"):
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        labels_dict.update(data)
                except Exception as e:
                    print(f"Failed to parse {json_path}: {e}")
    

    if vid_path.is_file():
        print(f"Processing single video: {vid_path}")
        extractor.process_video(vid_path, args.label, args.dataset)
    elif vid_path.is_dir():
        videos = list(vid_path.glob("*.mp4"))
        print(f"Found {len(videos)} videos.")
        for v in tqdm(videos, desc="Extracting"):
            video_name = v.name
            target_label = args.label
            if video_name in labels_dict:
                label_str = labels_dict[video_name].get("label", "").upper()
                if label_str == "FAKE":
                    target_label = 1
                elif label_str == "REAL":
                    target_label = 0
            extractor.process_video(v, target_label, args.dataset)
