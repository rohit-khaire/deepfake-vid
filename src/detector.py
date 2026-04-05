import torch
import torch.nn.functional as F
from mesonet import Meso4, load_pretrained_weights
from src.video_processor import extract_frames, detect_faces_in_frames, preprocess_frames_for_model
from typing import Dict, List
import numpy as np

class DeepfakeVideoDetector:
    def __init__(self, weights_path: str = None, threshold: float = 0.5):
        self.model = Meso4()
        if weights_path:
            load_pretrained_weights(self.model, weights_path)
        self.model.eval()
        self.threshold = threshold
    
    def detect_video(self, video_path: str) -> Dict:
        """Detect deepfakes in a video file."""
        # Extract frames
        frames, fps = extract_frames(video_path)
        if not frames:
            return {'error': 'No frames extracted from video'}
        
        # Detect faces
        face_frames = detect_faces_in_frames(frames)
        if not face_frames:
            return {'error': 'No faces detected in video'}
        
        # Preprocess
        input_tensor = preprocess_frames_for_model(face_frames)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            fake_probs = probabilities[:, 1].cpu().numpy()
        
        # Aggregate results
        avg_fake_prob = np.mean(fake_probs)
        max_fake_prob = np.max(fake_probs)
        min_fake_prob = np.min(fake_probs)
        
        is_fake = avg_fake_prob > self.threshold
        
        return {
            'is_fake': is_fake,
            'average_fake_probability': float(avg_fake_prob),
            'max_fake_probability': float(max_fake_prob),
            'min_fake_probability': float(min_fake_prob),
            'frame_predictions': fake_probs.tolist(),
            'num_frames_analyzed': len(fake_probs),
            'fps': fps,
            'threshold': self.threshold
        }