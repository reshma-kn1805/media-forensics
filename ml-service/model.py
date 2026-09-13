import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class DeepfakeClassifier(nn.Module):
    """
    Deepfake image classifier using EfficientNet-B0.

    Classes:
        0 -> Real
        1 -> Fake
    """

    def __init__(self):
        super().__init__()

        # Load EfficientNet-B0 with ImageNet pretrained weights
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)

        # Get the number of features from the original classifier
        in_features = self.backbone.classifier[1].in_features

        # Replace the original ImageNet classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.backbone(x)

    def target_layer(self):
        """
        Returns the final convolutional layer.
        This layer can later be used for Grad-CAM explainability.
        """
        return self.backbone.features[-1]


# Class names used by the model
CLASS_NAMES = ["real", "fake"]


# Image size expected by EfficientNet-B0
IMAGE_SIZE = 224


# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_model():
    """
    Create and return a DeepfakeClassifier.
    """
    return DeepfakeClassifier()


if __name__ == "__main__":
    # Simple test
    model = create_model()

    # Create a dummy image
    test_image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    # Run the image through the model
    output = model(test_image)

    print("Model created successfully!")
    print("Input shape:", test_image.shape)
    print("Output shape:", output.shape)
    print("Class names:", CLASS_NAMES)