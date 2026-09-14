import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class DeepfakeClassifier(nn.Module):
    """
    Deepfake image classifier using EfficientNet-B0.

    Classes:
        0 -> Real
        1 -> Fake
    """

    def __init__(self):
        super().__init__()

        # Create EfficientNet-B0 architecture WITHOUT downloading
        # ImageNet pretrained weights.
        #
        # The trained VERITAS checkpoint provides the actual weights
        # during inference.
        self.backbone = efficientnet_b0(weights=None)

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
        This layer is used for Grad-CAM explainability.
        """
        return self.backbone.features[-1]


# Default class names.
#
# The actual class mapping used during inference is read from
# the trained checkpoint by ml-service/app.py.
CLASS_NAMES = ["real", "fake"]


# Image size expected by EfficientNet-B0
IMAGE_SIZE = 224


# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_model():
    """
    Create and return a DeepfakeClassifier.

    No external model weights are downloaded here.
    The trained VERITAS checkpoint is loaded separately.
    """
    return DeepfakeClassifier()


if __name__ == "__main__":
    # Simple local architecture test
    model = create_model()

    # Create a dummy image
    test_image = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    # Run the image through the model
    output = model(test_image)

    print("Model architecture created successfully!")
    print("Input shape:", test_image.shape)
    print("Output shape:", output.shape)
    print("Class names:", CLASS_NAMES)
    print("Pretrained download required: No")