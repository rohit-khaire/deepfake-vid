import numpy as np
from typing import List, Tuple
import torch
from preprocess import preprocess_image_from_array

def extract_frames(video_path: str, fps: int = 30, max_frames: int = 300) -> Tuple[List[np.ndarray], float]:
    """Extract frames from video at specified FPS."""
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV not available. Install opencv-python.")
    
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / fps) if fps < original_fps else 1
    
    frames = []
    frame_count = 0
    
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        frame_count += 1
    
    cap.release()
    return frames, original_fps

def detect_faces_in_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Resize frames to standard size (skip face detection for now)."""
    try:
        import cv2
    except ImportError:
        # Fallback: just use numpy resize
        resized_frames = []
        for frame in frames:
            resized = np.array(frame)  # assuming already numpy
            resized_frames.append(cv2.resize(resized, (256, 256)) if 'cv2' in globals() else resized)
        return resized_frames
    
    resized_frames = []
    for frame in frames:
        resized = cv2.resize(frame, (256, 256))
        resized_frames.append(resized)
    return resized_frames

def preprocess_frames_for_model(frames: List[np.ndarray]) -> torch.Tensor:
    """Preprocess face frames for the model."""
    tensors = []
    for frame in frames:
        tensor = preprocess_image_from_array(frame)
        tensors.append(tensor)
    
    return torch.cat(tensors, dim=0)