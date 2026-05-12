<div align="center">
  <h1>🕵️‍♂️ Fake Video Detection</h1>
  <p><i>A robust, multimodal deepfake detection system leveraging Vision Transformers, Audio analysis, and Temporal modeling.</i></p>

  ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
  ![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
  ![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
  ![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
</div>

---

## 📖 Overview

As deepfake technology becomes more sophisticated, single-modality detection systems are no longer sufficient. This project implements a **state-of-the-art multimodal deepfake detection architecture**. By analyzing inconsistencies across different modalities—audio, frequency (visual artifacts), and temporal dynamics—the system achieves highly accurate detection of manipulated and synthetic videos.

## ✨ Key Features

- **Multimodal Fusion**: Combines spatial (frequency), temporal, and audio features for robust decision making.
- **Vision Transformers (ViT)**: Utilizes powerful frame-level feature extraction for high-resolution artifact detection.
- **Temporal Consistency**: Detects unnatural flickers and inter-frame manipulations.
- **Audio-Visual Desync**: Identifies mismatched audio and lip movements common in deepfakes.
- **Class-Imbalance Handling**: Employs weighted sampling during training for optimized real-world performance.

---

## 🚀 Model Performance

The final multimodal model demonstrates strong generalization and detection capabilities. Based on our latest large-scale training job (`train_job.o18446` on NVIDIA H100 GPUs), the model achieved the following performance metrics on the validation set:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | `89.20%` |
| **F1 Score** | `0.9391` |
| **AUC ROC** | `0.8358` |
| **Validation Loss** | `0.4979` |

> [!NOTE]
> *The model prioritizes **F1 Score** and **AUC** metrics over pure accuracy to account for heavy class imbalances (e.g., 88.6% majority class in training data).*

---

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/FakeVideoDetection.git
   cd FakeVideoDetection
   ```

2. **Install dependencies**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Inference/Deployment**
   Run the deployment script to test a sample video:
   ```bash
   python src/deploy.py --video_path path/to/video.mp4
   ```

---

## 🔮 Future Work

Our development roadmap includes several key milestones to further enhance the system's robustness:

### 🧩 Ablation Study
To rigorously understand the contribution of each component within our multimodal architecture, we will conduct a comprehensive ablation study:
- **Component Isolation**: Evaluating the system without individual parts (e.g., completely removing the audio stream, frequency stream, or temporal module).
- **Cross-Modality Testing**: Testing different combinations of modalities against one another (e.g., *Audio + Frequency* vs. *Frequency + Temporal*).

*This will help identify the most critical parts of the network and guide optimizations to balance computational efficiency with detection accuracy.*

### 🧪 Extensive Testing
- **Unit and Integration Testing**: Implementing a rigorous test suite for all preprocessing pipelines (e.g., MTCNN face extraction, Librosa audio processing) and model fusion components to guarantee pipeline stability.
- **Adversarial Testing**: Evaluating the model against known adversarial attacks designed specifically to bypass standard deepfake detectors.

### 🌍 Generalization Testing
- **Cross-Dataset Evaluation**: Testing the model on unseen, completely distinct datasets (e.g., evaluating on Celeb-DF or Eval 2024 when trained on DFDC) to measure real-world zero-shot generalization.
- **In-the-Wild Robustness**: Evaluating performance on heavily compressed videos, low-lighting scenarios, and varied resolutions to ensure reliability in unconstrained, real-world environments.
