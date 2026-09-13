import base64
import io

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def generate_gradcam(
    model,
    input_tensor: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    """
    Generate a Grad-CAM grayscale heatmap for the selected class.

    Grad-CAM requires gradients, so this function must NOT be called
    inside a torch.no_grad() block.
    """
    model.eval()

    target_layers = [model.target_layer()]

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
    )

    targets = [ClassifierOutputTarget(int(target_class))]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets,
    )

    return grayscale_cam[0]


def create_gradcam_overlay(
    face_image: Image.Image,
    grayscale_cam: np.ndarray,
) -> Image.Image:
    """
    Create a visual Grad-CAM overlay on the analyzed face image.
    """
    face_rgb = face_image.convert("RGB")
    image_array = np.asarray(face_rgb)

    grayscale_cam = cv2.resize(
        grayscale_cam,
        (image_array.shape[1], image_array.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    rgb_float = image_array.astype(np.float32) / 255.0

    overlay = show_cam_on_image(
        rgb_float,
        grayscale_cam,
        use_rgb=True,
    )

    return Image.fromarray(overlay)


def gradcam_to_base64(
    overlay_image: Image.Image,
    image_format: str = "PNG",
) -> str:
    """
    Convert the Grad-CAM overlay to a base64 string suitable for
    returning through the FastAPI JSON response.
    """
    buffer = io.BytesIO()

    overlay_image.save(
        buffer,
        format=image_format,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return encoded


def generate_gradcam_base64(
    model,
    input_tensor: torch.Tensor,
    face_image: Image.Image,
    target_class: int,
) -> str:
    """
    Complete Grad-CAM pipeline:
        model + input -> heatmap -> overlay -> base64 PNG
    """
    grayscale_cam = generate_gradcam(
        model=model,
        input_tensor=input_tensor,
        target_class=target_class,
    )

    overlay = create_gradcam_overlay(
        face_image=face_image,
        grayscale_cam=grayscale_cam,
    )

    return gradcam_to_base64(
        overlay_image=overlay,
        image_format="PNG",
    )
