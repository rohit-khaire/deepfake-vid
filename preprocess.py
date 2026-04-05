import torch
from torchvision import transforms
from PIL import Image
import numpy as np


def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess an image for MesoNet deepfake detection.

    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Target size to resize the image (height, width).

    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input.
    """
    # Open image
    image = Image.open(image_path).convert('RGB')

    # Define preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Apply transforms
    image_tensor = preprocess(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def preprocess_image_from_array(image_array, target_size=(256, 256)):
    """
    Preprocess an image array (numpy array) for MesoNet deepfake detection.

    Args:
        image_array (np.ndarray): Input image as numpy array (H, W, C) or (C, H, W).
        target_size (tuple): Target size to resize the image (height, width).

    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input.
    """
    # Convert to PIL Image if necessary
    if isinstance(image_array, np.ndarray):
        if image_array.shape[-1] == 3:  # HWC format
            image = Image.fromarray(image_array.astype('uint8'), 'RGB')
        else:  # Assume CHW format
            image_array = np.transpose(image_array, (1, 2, 0))
            image = Image.fromarray(image_array.astype('uint8'), 'RGB')
    else:
        raise ValueError("Input must be a numpy array")

    # Define preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transforms
    image_tensor = preprocess(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


# Example usage
if __name__ == "__main__":
    # Test with a dummy image path (replace with actual path)
    # tensor = preprocess_image("path/to/image.jpg")
    # print(f"Preprocessed tensor shape: {tensor.shape}")

    # Test with numpy array
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    tensor = preprocess_image_from_array(dummy_image)
    print(f"Preprocessed tensor shape: {tensor.shape}")