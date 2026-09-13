from __future__ import annotations

import hashlib
import io
import math
from typing import Any

import cv2
import numpy as np
from PIL import ExifTags, Image


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_number(value: Any) -> float | None:
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass

    return None


def metadata_analysis(image: Image.Image) -> dict[str, Any]:
    exif = image.getexif()
    mapped: dict[str, Any] = {}

    if exif:
        for key, value in exif.items():
            name = ExifTags.TAGS.get(key, str(key))

            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", errors="replace")
                except Exception:
                    value = "<binary>"

            mapped[name] = str(value)

    gps_present = bool(mapped.get("GPSInfo"))

    return {
        "format": image.format or "UNKNOWN",
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "exif_present": bool(mapped),
        "exif_status": "PRESENT" if mapped else "NOT PRESENT",
        "camera_make": mapped.get("Make", "Not available"),
        "camera_model": mapped.get("Model", "Not available"),
        "date_time": mapped.get(
            "DateTime",
            mapped.get("DateTimeOriginal", "Not available"),
        ),
        "software": mapped.get("Software", "Not available"),
        "orientation": mapped.get("Orientation", "Not available"),
        "gps_present": gps_present,
        "gps_status": "PRESENT" if gps_present else "NOT PRESENT",
        "raw": mapped,
    }


def integrity_analysis(
    data: bytes,
    image: Image.Image,
) -> dict[str, Any]:

    digest = sha256_bytes(data)
    format_name = image.format or "UNKNOWN"

    decode_check = "PASSED"

    try:
        with Image.open(io.BytesIO(data)) as check:
            check.load()
    except Exception:
        decode_check = "FAILED"

    return {
        "status": "PASS" if decode_check == "PASSED" else "FAIL",
        "format": format_name,
        "file_size_bytes": len(data),
        "file_size_kb": round(len(data) / 1024, 2),
        "sha256": digest,
        "verification": (
            "PASSED"
            if decode_check == "PASSED"
            else "FAILED"
        ),
        "decode_check": decode_check,
    }


def compression_analysis(
    data: bytes,
    image: Image.Image,
) -> dict[str, Any]:

    pixels = max(1, image.width * image.height)
    bytes_per_pixel = len(data) / pixels

    quantization_tables = bool(
        getattr(image, "quantization", None)
    )

    if image.format == "JPEG":

        if bytes_per_pixel < 0.08:
            level = "Very High"

        elif bytes_per_pixel < 0.16:
            level = "High"

        elif bytes_per_pixel < 0.35:
            level = "Moderate"

        else:
            level = "Low"

    else:
        level = "Not applicable"

    return {
        "status": "PASS",
        "format": image.format or "UNKNOWN",
        "compression_level": level,
        "bytes_per_pixel": round(bytes_per_pixel, 4),
        "pixel_count": pixels,
        "quantization_tables_present": quantization_tables,
        "decode_check": "PASSED",
    }


def noise_frequency_analysis(
    image: Image.Image,
) -> dict[str, Any]:

    rgb = np.asarray(
        image.convert("RGB")
    )

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    ).astype(np.float32)

    # Noise estimation
    median = cv2.medianBlur(
        gray.astype(np.uint8),
        3,
    ).astype(np.float32)

    residual = gray - median

    noise_variance = float(
        np.var(residual)
    )

    # Sharpness estimation
    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    # Frequency-domain analysis
    spectrum = np.fft.fftshift(
        np.fft.fft2(gray)
    )

    magnitude = np.abs(spectrum)

    height, width = gray.shape

    center_y = height // 2
    center_x = width // 2

    radius = max(
        2,
        int(min(height, width) * 0.08),
    )

    yy, xx = np.ogrid[
        :height,
        :width,
    ]

    low_frequency_mask = (
        (yy - center_y) ** 2
        + (xx - center_x) ** 2
        <= radius ** 2
    )

    energy = magnitude ** 2

    total_energy = float(
        np.sum(energy)
    )

    low_energy = float(
        np.sum(
            energy[low_frequency_mask]
        )
    )

    if total_energy <= 0:
        high_frequency_ratio = 0.0

    else:
        high_frequency_ratio = max(
            0.0,
            min(
                1.0,
                (
                    total_energy
                    - low_energy
                )
                / total_energy,
            ),
        )

    def level(
        value: float,
        low: float,
        high: float,
    ) -> str:

        if value < low:
            return "Low"

        if value < high:
            return "Moderate"

        return "High"

    return {
        "status": "PASS",
        "noise_level": level(
            noise_variance,
            10.0,
            100.0,
        ),
        "noise_variance": round(
            noise_variance,
            4,
        ),
        "sharpness": round(
            sharpness,
            4,
        ),
        "high_frequency_ratio": round(
            high_frequency_ratio * 100,
            2,
        ),
        "frequency_profile": (
            "High-frequency dominant"
            if high_frequency_ratio >= 0.5
            else "Low-frequency dominant"
        ),
    }


def run_forensics(
    data: bytes,
    image: Image.Image,
) -> dict[str, Any]:

    metadata = metadata_analysis(image)

    integrity = integrity_analysis(
        data,
        image,
    )

    compression = compression_analysis(
        data,
        image,
    )

    noise_frequency = (
        noise_frequency_analysis(image)
    )

    return {
        "metadata": metadata,
        "integrity": integrity,
        "compression": compression,
        "noise_frequency": noise_frequency,
    }