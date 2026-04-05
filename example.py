import torch
import torch.nn.functional as F
from mesonet import Meso4, load_pretrained_weights
from preprocess import preprocess_image
import argparse


def predict_deepfake(model, image_tensor, device='cpu'):
    """
    Predict whether an image is a deepfake or real.

    Args:
        model (nn.Module): The trained MesoNet model.
        image_tensor (torch.Tensor): Preprocessed image tensor.
        device (str): Device to run inference on ('cpu' or 'cuda').

    Returns:
        dict: Prediction results with probabilities and class.
    """
    model.to(device)
    model.eval()

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    class_names = ['real', 'fake']
    predicted_class = class_names[predicted.item()]
    confidence = probabilities[0][predicted.item()].item()

    return {
        'prediction': predicted_class,
        'confidence': confidence,
        'probabilities': {
            'real': probabilities[0][0].item(),
            'fake': probabilities[0][1].item()
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Detect deepfakes using MesoNet')
    parser.add_argument('image_path', type=str, help='Path to the image to analyze')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to pre-trained weights file (.pth)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'], help='Device to run inference on')

    args = parser.parse_args()

    # Initialize the model
    model = Meso4()

    # Load pre-trained weights if provided
    if args.weights:
        model = load_pretrained_weights(model, args.weights)
        print(f"Loaded pre-trained weights from {args.weights}")
    else:
        print("Warning: No pre-trained weights loaded. Using randomly initialized model.")
        print("For accurate results, download pre-trained weights from the MesoNet repository.")

    # Preprocess the image
    image_tensor = preprocess_image(args.image_path)
    print(f"Preprocessed image shape: {image_tensor.shape}")

    # Make prediction
    result = predict_deepfake(model, image_tensor, args.device)

    # Print results
    print("\nDeepfake Detection Results:")
    print(f"Prediction: {result['prediction']}")
    print(".3f")
    print(".3f")
    print(".3f")


if __name__ == "__main__":
    main()