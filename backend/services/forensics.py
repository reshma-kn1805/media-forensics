from __future__ import annotations

import hashlib
import io
import math
from typing import Any

from PIL import ExifTags, Image

from backend.services.noise_frequency_analysis import (
    analyze_noise_frequency,
)


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


def metadata_analysis(
    image: Image.Image,
) -> dict[str, Any]:

    exif = image.getexif()

    mapped: dict[str, Any] = {}

    if exif:
        for key, value in exif.items():

            name = ExifTags.TAGS.get(
                key,
                str(key),
            )

            if isinstance(value, bytes):

                try:
                    value = value.decode(
                        "utf-8",
                        errors="replace",
                    )

                except Exception:
                    value = "<binary>"

            mapped[name] = str(value)

    gps_present = bool(
        mapped.get("GPSInfo")
    )

    return {
        "format": image.format or "UNKNOWN",
        "width": image.width,
        "height": image.height,
        "mode": image.mode,

        "exif_present": bool(mapped),

        "exif_status": (
            "PRESENT"
            if mapped
            else "NOT PRESENT"
        ),

        "camera_make": mapped.get(
            "Make",
            "Not available",
        ),

        "camera_model": mapped.get(
            "Model",
            "Not available",
        ),

        "date_time": mapped.get(
            "DateTime",
            mapped.get(
                "DateTimeOriginal",
                "Not available",
            ),
        ),

        "software": mapped.get(
            "Software",
            "Not available",
        ),

        "orientation": mapped.get(
            "Orientation",
            "Not available",
        ),

        "gps_present": gps_present,

        "gps_status": (
            "PRESENT"
            if gps_present
            else "NOT PRESENT"
        ),

        "raw": mapped,
    }


def integrity_analysis(
    data: bytes,
    image: Image.Image,
) -> dict[str, Any]:

    digest = sha256_bytes(data)

    format_name = (
        image.format
        or "UNKNOWN"
    )

    decode_check = "PASSED"

    try:

        with Image.open(
            io.BytesIO(data)
        ) as check:

            check.load()

    except Exception:

        decode_check = "FAILED"

    return {
        "status": (
            "PASS"
            if decode_check == "PASSED"
            else "FAIL"
        ),

        "format": format_name,

        "file_size_bytes": len(data),

        "file_size_kb": round(
            len(data) / 1024,
            2,
        ),

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

    pixels = max(
        1,
        image.width * image.height,
    )

    bytes_per_pixel = (
        len(data) / pixels
    )

    quantization_tables = bool(
        getattr(
            image,
            "quantization",
            None,
        )
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

        "format": (
            image.format
            or "UNKNOWN"
        ),

        "compression_level": level,

        "bytes_per_pixel": round(
            bytes_per_pixel,
            4,
        ),

        "pixel_count": pixels,

        "quantization_tables_present": (
            quantization_tables
        ),

        "decode_check": "PASSED",
    }


def noise_frequency_analysis(
    image: Image.Image,
) -> dict[str, Any]:
    """
    Run the complete Noise & Frequency forensic module.

    The actual implementation is centralized in
    backend.services.noise_frequency_analysis.

    This wrapper is intentionally kept here so that
    run_forensics() exposes the complete result through
    the main /analyze endpoint.
    """

    result = analyze_noise_frequency(
        image
    )

    if not isinstance(result, dict):
        raise ValueError(
            "Noise & frequency analysis "
            "returned an invalid result."
        )

    return result


def run_forensics(
    data: bytes,
    image: Image.Image,
) -> dict[str, Any]:
    """
    Run all non-ML forensic modules.

    Noise & Frequency Analysis is delegated to the
    dedicated forensic service so that the complete
    visual analysis and metrics are returned.
    """

    metadata = metadata_analysis(
        image
    )

    integrity = integrity_analysis(
        data,
        image,
    )

    compression = compression_analysis(
        data,
        image,
    )

    noise_frequency = (
        noise_frequency_analysis(
            image
        )
    )

    return {
        "metadata": metadata,

        "integrity": integrity,

        "compression": compression,

        "noise_frequency": (
            noise_frequency
        ),
    }