import io

import numpy as np
from PIL import Image


def analyze_compression(image_bytes: bytes, image: Image.Image) -> dict:
    """
    Analyze basic compression and encoding characteristics of an image.

    These measurements are supporting forensic indicators only.
    They do not independently prove whether an image is authentic
    or manipulated.
    """

    file_size_bytes = len(image_bytes)

    image_format = image.format or "Unknown"
    width, height = image.size

    pixel_count = int(width) * int(height)

    bytes_per_pixel = (
        file_size_bytes / pixel_count
        if pixel_count > 0
        else 0.0
    )

    # ---------------------------------------------------------
    # Decode verification
    # ---------------------------------------------------------

    decode_check_passed = False

    try:
        verification_image = Image.open(
            io.BytesIO(image_bytes)
        )

        verification_image.load()

        decode_check_passed = True

    except Exception:
        decode_check_passed = False

    # ---------------------------------------------------------
    # Progressive JPEG detection
    # ---------------------------------------------------------

    progressive = False

    if image_format.upper() == "JPEG":
        try:
            progressive = bool(
                image.info.get("progressive", False)
                or image.info.get("progression", False)
            )
        except Exception:
            progressive = False

    # ---------------------------------------------------------
    # Compression estimation
    # ---------------------------------------------------------

    format_upper = image_format.upper()

    if format_upper == "JPEG":

        if bytes_per_pixel < 0.08:
            compression_level = "Very high"

        elif bytes_per_pixel < 0.18:
            compression_level = "High"

        elif bytes_per_pixel < 0.35:
            compression_level = "Moderate"

        else:
            compression_level = "Low"

    elif format_upper == "WEBP":

        if bytes_per_pixel < 0.08:
            compression_level = "Very high"

        elif bytes_per_pixel < 0.18:
            compression_level = "High"

        elif bytes_per_pixel < 0.35:
            compression_level = "Moderate"

        else:
            compression_level = "Low"

    elif format_upper == "PNG":

        # PNG is lossless, so the file may still be highly
        # compressed at the encoding level, but it is not
        # lossy compression in the same sense as JPEG.
        compression_level = "Lossless"

    else:
        compression_level = "Unknown"

    # ---------------------------------------------------------
    # Estimated quality description
    # ---------------------------------------------------------

    if format_upper in {"JPEG", "WEBP"}:

        if bytes_per_pixel < 0.08:
            quality_estimate = "Strong compression"

        elif bytes_per_pixel < 0.18:
            quality_estimate = "Compressed"

        elif bytes_per_pixel < 0.35:
            quality_estimate = "Moderately compressed"

        else:
            quality_estimate = "Lightly compressed"

    elif format_upper == "PNG":
        quality_estimate = "Lossless encoding"

    else:
        quality_estimate = "Not available"

    # ---------------------------------------------------------
    # Basic consistency checks
    # ---------------------------------------------------------

    dimensions_valid = (
        width > 0
        and height > 0
    )

    file_size_valid = file_size_bytes > 0

    format_supported = format_upper in {
        "JPEG",
        "PNG",
        "WEBP",
    }

    analysis_passed = (
        dimensions_valid
        and file_size_valid
        and decode_check_passed
        and format_supported
    )

    status = "PASS" if analysis_passed else "WARNING"

    # ---------------------------------------------------------
    # Warnings
    # ---------------------------------------------------------

    warnings = []

    if not dimensions_valid:
        warnings.append(
            "Invalid image dimensions detected."
        )

    if not file_size_valid:
        warnings.append(
            "The submitted file contains no data."
        )

    if not decode_check_passed:
        warnings.append(
            "The image could not be fully decoded."
        )

    if not format_supported:
        warnings.append(
            "The image format is outside the currently supported forensic formats."
        )

    if format_upper == "JPEG" and progressive:
        warnings.append(
            "The JPEG image uses progressive encoding."
        )

    if format_upper in {"JPEG", "WEBP"} and bytes_per_pixel < 0.08:
        warnings.append(
            "The image has a relatively low byte-per-pixel ratio, "
            "indicating strong compression."
        )

    # ---------------------------------------------------------
    # Return forensic analysis
    # ---------------------------------------------------------

    return {
        "status": status,

        "format": image_format,

        "file_size_bytes": int(file_size_bytes),

        "file_size_kb": round(
            file_size_bytes / 1024,
            2,
        ),

        "image_width": int(width),

        "image_height": int(height),

        "pixel_count": pixel_count,

        "bytes_per_pixel": round(
            bytes_per_pixel,
            4,
        ),

        "compression_level": compression_level,

        "quality_estimate": quality_estimate,

        "progressive": progressive,

        "decode_check_passed": decode_check_passed,

        "dimensions_valid": dimensions_valid,

        "file_size_valid": file_size_valid,

        "format_supported": format_supported,

        "warnings": warnings,

        "forensic_note": (
            "Compression analysis describes encoding and file-size "
            "characteristics of the submitted image. Compression "
            "characteristics are supporting forensic indicators only "
            "and do not independently establish whether an image is "
            "authentic or manipulated."
        ),
    }