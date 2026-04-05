import torch
import torch.nn as nn
import torch.nn.functional as F


class Meso4(nn.Module):
    """
    MesoNet Meso-4 architecture for deepfake detection.
    Based on the paper: "MesoNet: a Compact Facial Video Forgery Detection Network"
    by Darius Afchar et al.
    """

    def __init__(self, num_classes=2):
        super(Meso4, self).__init__()
        self.num_classes = num_classes

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv6 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv7 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv8 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.conv9 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv10 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv11 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)

        self.conv13 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv14 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv15 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv16 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2)

        self.conv17 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv18 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2)

        self.conv19 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv20 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool6 = nn.MaxPool2d(2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional blocks with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool1(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool2(x)

        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool3(x)

        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        x = self.pool4(x)

        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))
        x = self.pool5(x)

        x = F.relu(self.conv19(x))
        x = F.relu(self.conv20(x))
        x = self.pool6(x)

        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def load_pretrained_weights(model, weights_path):
    """
    Load pre-trained weights into the model.

    Args:
        model (nn.Module): The MesoNet model instance.
        weights_path (str): Path to the pre-trained weights file (.pth or .pt).

    Returns:
        nn.Module: The model with loaded weights, set to evaluation mode.
    """
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model


# Example of creating the model
if __name__ == "__main__":
    model = Meso4()
    print(model)
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be (1, 2)