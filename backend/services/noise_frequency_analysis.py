import base64

import cv2
import numpy as np
from PIL import Image


def image_to_base64(
    image: np.ndarray,
    color_mode: str = "BGR",
) -> str:
    """Convert an OpenCV image into a base64 PNG string."""
    if color_mode == "GRAY":
        output_image = image
    elif color_mode == "RGB":
        output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        output_image = image

    success, encoded = cv2.imencode(".png", output_image)
    if not success:
        raise ValueError("Unable to encode analysis image.")

    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize an array safely into the 0-255 uint8 range."""
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    minimum = float(np.min(image))
    maximum = float(np.max(image))

    if maximum - minimum < 1e-8:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = (image - minimum) / (maximum - minimum)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def create_low_frequency_component(gray_image: np.ndarray) -> np.ndarray:
    """
    Generate the low-frequency image component.

    Low-frequency content represents the smooth, large-scale structures
    of the submitted image after suppressing fine detail and noise.
    """
    low_frequency = cv2.GaussianBlur(
        gray_image,
        (21, 21),
        0,
    )
    return low_frequency.astype(np.uint8)


def create_high_frequency_component(gray_image: np.ndarray) -> np.ndarray:
    """
    Generate the high-frequency image component.

    This isolates edges, fine texture and high-frequency residual detail
    by subtracting a low-pass representation from the original image.
    """
    low_frequency = create_low_frequency_component(gray_image)
    residual = (
        gray_image.astype(np.float32)
        - low_frequency.astype(np.float32)
    )

    # Center the residual around mid-gray so positive and negative
    # high-frequency components remain visible in the final image.
    centered = residual + 128.0
    return np.clip(centered, 0, 255).astype(np.uint8)


def create_noise_map(gray_image: np.ndarray) -> np.ndarray:
    """Generate a visual high-frequency noise map."""
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    residual = cv2.absdiff(gray_image, blurred)
    normalized = normalize_to_uint8(residual)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def create_noise_heatmap(gray_image: np.ndarray) -> np.ndarray:
    """Generate a smoothed heatmap showing spatial noise concentration."""
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    residual = cv2.absdiff(gray_image, blurred)
    residual = cv2.GaussianBlur(residual, (11, 11), 0)
    normalized = normalize_to_uint8(residual)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def create_fft_spectrum(gray_image: np.ndarray) -> np.ndarray:
    """Generate a visual FFT frequency spectrum."""
    float_image = gray_image.astype(np.float32)
    float_image -= np.mean(float_image)

    fft = np.fft.fft2(float_image)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    log_magnitude = np.log1p(magnitude)

    normalized = normalize_to_uint8(log_magnitude)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def calculate_frequency_distribution(
    log_magnitude: np.ndarray,
    bins: int = 50,
) -> list:
    """Generate normalized low-to-high frequency distribution data."""
    height, width = log_magnitude.shape
    center_y = height // 2
    center_x = width // 2

    y, x = np.ogrid[:height, :width]
    distance = np.sqrt(
        (x - center_x) ** 2
        + (y - center_y) ** 2
    )

    max_radius = float(distance.max())
    if max_radius <= 0:
        return []

    radius_values = distance / max_radius
    distribution = []

    for index in range(bins):
        start = index / bins
        end = (index + 1) / bins
        mask = (radius_values >= start) & (radius_values < end)
        values = log_magnitude[mask]
        energy = float(np.mean(values)) if values.size else 0.0

        distribution.append(
            {
                "frequency_radius": round((start + end) / 2, 4),
                "energy": round(energy, 6),
            }
        )

    return distribution


def analyze_noise_frequency(image: Image.Image) -> dict:
    """
    Perform complete spatial-noise and frequency-domain analysis.

    All visual outputs are generated from the actual submitted image.
    These measurements are supporting forensic indicators only.
    """
    rgb_image = np.array(image.convert("RGB"))
    if rgb_image.size == 0:
        raise ValueError("The image contains no pixel data.")

    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    height, width = gray_image.shape

    if height < 8 or width < 8:
        raise ValueError(
            "Image is too small for reliable noise and frequency analysis."
        )

    # ---------------------------------------------------------
    # 1. Noise residual
    # ---------------------------------------------------------
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    noise_residual = (
        gray_image.astype(np.float32)
        - blurred.astype(np.float32)
    )

    noise_variance = float(np.var(noise_residual))
    noise_standard_deviation = float(np.std(noise_residual))

    # ---------------------------------------------------------
    # 2. Sharpness
    # ---------------------------------------------------------
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    sharpness_variance = float(laplacian.var())

    # ---------------------------------------------------------
    # 3. FFT
    # ---------------------------------------------------------
    float_gray = gray_image.astype(np.float32)
    float_gray -= np.mean(float_gray)

    fft = np.fft.fft2(float_gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    log_magnitude = np.log1p(magnitude)

    frequency_energy = float(np.mean(log_magnitude))

    # ---------------------------------------------------------
    # 4. High-frequency ratio
    # ---------------------------------------------------------
    center_y = height // 2
    center_x = width // 2

    y, x = np.ogrid[:height, :width]
    distance = np.sqrt(
        (x - center_x) ** 2
        + (y - center_y) ** 2
    )

    low_frequency_radius = min(center_x, center_y) * 0.25
    high_frequency_mask = distance > low_frequency_radius

    total_energy = float(np.sum(log_magnitude))
    high_frequency_energy = float(
        np.sum(log_magnitude[high_frequency_mask])
    )

    if total_energy > 0:
        high_frequency_ratio = high_frequency_energy / total_energy
    else:
        high_frequency_ratio = 0.0

    high_frequency_ratio = float(
        np.clip(high_frequency_ratio, 0.0, 1.0)
    )

    # ---------------------------------------------------------
    # 5. Qualitative values
    # ---------------------------------------------------------
    if noise_standard_deviation < 3:
        noise_level = "Low"
    elif noise_standard_deviation < 8:
        noise_level = "Moderate"
    else:
        noise_level = "High"

    if sharpness_variance < 50:
        sharpness_level = "Very low"
    elif sharpness_variance < 150:
        sharpness_level = "Low"
    elif sharpness_variance < 500:
        sharpness_level = "Moderate"
    elif sharpness_variance < 1500:
        sharpness_level = "High"
    else:
        sharpness_level = "Very high"

    if high_frequency_ratio < 0.65:
        frequency_characteristic = "Low-frequency dominant"
    elif high_frequency_ratio < 0.80:
        frequency_characteristic = "Balanced frequency distribution"
    else:
        frequency_characteristic = "High-frequency dominant"

    # ---------------------------------------------------------
    # 6. Visual forensic components
    # ---------------------------------------------------------
    low_frequency_component = create_low_frequency_component(gray_image)
    high_frequency_component = create_high_frequency_component(gray_image)
    noise_map = create_noise_map(gray_image)
    fft_spectrum = create_fft_spectrum(gray_image)
    noise_heatmap = create_noise_heatmap(gray_image)
    frequency_distribution = calculate_frequency_distribution(log_magnitude)

    # ---------------------------------------------------------
    # 7. Warnings
    # ---------------------------------------------------------
    warnings = []

    if noise_level == "Low":
        warnings.append(
            "The image contains relatively low estimated high-frequency noise."
        )
    elif noise_level == "High":
        warnings.append(
            "The image contains relatively high estimated high-frequency noise."
        )

    if frequency_characteristic == "High-frequency dominant":
        warnings.append(
            "The frequency spectrum contains a relatively high proportion of high-frequency energy."
        )

    return {
        "status": "PASS",
        "image_width": int(width),
        "image_height": int(height),
        "noise_level": noise_level,
        "noise_variance": round(noise_variance, 4),
        "noise_standard_deviation": round(noise_standard_deviation, 4),
        "sharpness_variance": round(sharpness_variance, 4),
        "sharpness_level": sharpness_level,
        "frequency_energy": round(frequency_energy, 4),
        "high_frequency_energy_ratio": round(high_frequency_ratio, 4),
        "frequency_characteristic": frequency_characteristic,
        "low_frequency_base64": image_to_base64(
            low_frequency_component,
            color_mode="GRAY",
        ),
        "high_frequency_base64": image_to_base64(
            high_frequency_component,
            color_mode="GRAY",
        ),
        "noise_map_base64": image_to_base64(noise_map),
        "fft_spectrum_base64": image_to_base64(fft_spectrum),
        "noise_heatmap_base64": image_to_base64(noise_heatmap),
        "frequency_distribution": frequency_distribution,
        "warnings": warnings,
        "forensic_note": (
            "Noise, sharpness, and frequency characteristics are supporting "
            "forensic indicators only. They do not independently establish "
            "whether an image is authentic or manipulated."
        ),
    }
