from PIL import Image
from torchvision import transforms


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# ============================================================
# IMAGE TRANSFORM
# ============================================================

def get_transform():
    """
    Create the preprocessing transformation used
    before sending an image to EfficientNet-B0.
    """

    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image: Image.Image):
    """
    Convert a PIL image into a model-ready tensor.

    Steps:
        1. Convert image to RGB.
        2. Resize to 224 x 224.
        3. Convert to PyTorch tensor.
        4. Normalize using ImageNet statistics.
        5. Add batch dimension.

    Returns:
        PyTorch tensor with shape:

        [1, 3, 224, 224]
    """

    image = image.convert("RGB")

    transform = get_transform()

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    return tensor