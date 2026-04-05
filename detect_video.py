#!/usr/bin/env python3
import argparse
import sys
from src.detector import DeepfakeVideoDetector

def main():
    parser = argparse.ArgumentParser(description='GOD Level Deepfake Video Detector')
    parser.add_argument('video_path', help='Path to the video file to analyze')
    parser.add_argument('--weights', help='Path to pre-trained model weights')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Threshold for classifying as fake (default: 0.5)')
    parser.add_argument('--output', help='Output file to save results (JSON)')
    
    args = parser.parse_args()
    
    detector = DeepfakeVideoDetector(weights_path=args.weights, threshold=args.threshold)
    result = detector.detect_video(args.video_path)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    
    print("Deepfake Video Detection Results:")
    print(f"Video: {args.video_path}")
    print(f"Average Fake Probability: {result['average_fake_probability']:.4f}")
    print(f"Max Fake Probability: {result['max_fake_probability']:.4f}")
    print(f"Min Fake Probability: {result['min_fake_probability']:.4f}")
    print(f"Frames Analyzed: {result['num_frames_analyzed']}")
    print(f"FPS: {result['fps']}")
    print(f"Verdict: {'FAKE' if result['is_fake'] else 'REAL'}")
    print(f"Threshold: {result['threshold']}")
    
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()