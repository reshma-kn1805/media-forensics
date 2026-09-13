import hashlib
import io

from PIL import Image


def analyze_image_integrity(
    image_bytes: bytes,
    image: Image.Image,
) -> dict:
    """
    Analyze basic image integrity and evidence properties.

    These checks describe the submitted file and its decoded
    image properties. They do NOT independently prove that an
    image is authentic or manipulated.
    """

    # =========================================================
    # 1. BASIC FILE INFORMATION
    # =========================================================

    file_size_bytes = len(image_bytes)

    image_format = (
        image.format or "Unknown"
    )

    width, height = image.size

    color_mode = image.mode

    has_alpha_channel = (
        "A" in image.getbands()
    )

    pixel_count = (
        int(width) * int(height)
    )

    # =========================================================
    # 2. SHA-256
    # =========================================================

    sha256_hash = hashlib.sha256(
        image_bytes
    ).hexdigest()

    # =========================================================
    # 3. EXIF
    # =========================================================

    try:
        exif = image.getexif()

        exif_field_count = len(exif)

        has_exif = (
            exif_field_count > 0
        )

    except Exception:
        exif_field_count = 0
        has_exif = False

    # =========================================================
    # 4. IMAGE VALIDATION
    # =========================================================

    dimensions_valid = (
        width > 0
        and height > 0
    )

    file_size_valid = (
        file_size_bytes > 0
    )

    decode_check_passed = False

    try:
        # Force a complete decode.
        image_copy = Image.open(
            io.BytesIO(image_bytes)
        )

        image_copy.load()

        decode_check_passed = True

    except Exception:
        decode_check_passed = False

    # =========================================================
    # 5. FORMAT VALIDATION
    # =========================================================

    supported_formats = {
        "JPEG",
        "PNG",
        "WEBP",
    }

    format_supported = (
        image_format.upper()
        in supported_formats
    )

    # =========================================================
    # 6. OVERALL STATUS
    # =========================================================

    verification_passed = (
        dimensions_valid
        and file_size_valid
        and decode_check_passed
        and format_supported
    )

    if verification_passed:
        status = "PASS"
    else:
        status = "WARNING"

    # =========================================================
    # 7. WARNINGS
    # =========================================================

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
            "The image format is outside the "
            "currently supported forensic formats."
        )

    if not has_exif:
        warnings.append(
            "No EXIF metadata was detected."
        )

    # =========================================================
    # 8. RESULT
    # =========================================================

    return {
        "status": status,

        "format": image_format,

        "file_size_bytes": file_size_bytes,

        "file_size_kb": round(
            file_size_bytes / 1024,
            2,
        ),

        "width": int(width),

        "height": int(height),

        "pixel_count": pixel_count,

        "color_mode": color_mode,

        "has_alpha_channel": (
            has_alpha_channel
        ),

        "has_exif": has_exif,

        "exif_field_count": (
            exif_field_count
        ),

        "dimensions_valid": (
            dimensions_valid
        ),

        "file_size_valid": (
            file_size_valid
        ),

        "decode_check_passed": (
            decode_check_passed
        ),

        "format_supported": (
            format_supported
        ),

        "image_verified": (
            verification_passed
        ),

        "sha256": sha256_hash,

        "warnings": warnings,

        "forensic_note": (
            "Image integrity checks describe the "
            "submitted file, its decoded properties, "
            "and available metadata. They do not "
            "independently establish whether an image "
            "is authentic or manipulated."
        ),
    }