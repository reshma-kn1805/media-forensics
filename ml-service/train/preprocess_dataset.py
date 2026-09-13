from pathlib import Path
import sys

import cv2
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ML_SERVICE_DIR = PROJECT_ROOT / "ml-service"

if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))


from utils.face_detector import create_face_detector


RAW_DATASET = PROJECT_ROOT / "data" / "raw" / "deepfake_dataset"
PROCESSED_DATASET = PROJECT_ROOT / "data" / "processed"

REAL_RAW = RAW_DATASET / "real"
FAKE_RAW = RAW_DATASET / "fake"

REAL_PROCESSED = PROCESSED_DATASET / "real"
FAKE_PROCESSED = PROCESSED_DATASET / "fake"


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

REAL_PROCESSED.mkdir(
    parents=True,
    exist_ok=True
)

FAKE_PROCESSED.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):
    """
    Load an image safely on Windows.

    This method supports Unicode characters in the
    project path.
    """

    try:

        image_bytes = np.fromfile(
            str(image_path),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR
        )

        return image

    except Exception as error:

        print(
            f"[ERROR] Failed to load "
            f"{image_path.name}: {error}"
        )

        return None


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(image, output_path):
    """
    Save an image safely on Windows.

    cv2.imwrite() can have problems with Unicode
    paths, so JPEG bytes are written using Python.
    """

    try:

        success, encoded_image = cv2.imencode(
            ".jpg",
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

        if not success:
            return False

        encoded_image.tofile(
            str(output_path)
        )

        return output_path.exists()

    except Exception as error:

        print(
            f"[ERROR] Failed to save "
            f"{output_path.name}: {error}"
        )

        return False


# ============================================================
# DETECT LARGEST FACE
# ============================================================

def detect_largest_face(image, detector):
    """
    Detect all faces and return the largest face.

    Returns:
        (x, y, width, height)

    or:

        None
    """

    faces = detector.detect_faces(image)

    if len(faces) == 0:
        return None

    largest_face = max(
        faces,
        key=lambda face: int(face[2]) * int(face[3])
    )

    return (
        int(largest_face[0]),
        int(largest_face[1]),
        int(largest_face[2]),
        int(largest_face[3])
    )


# ============================================================
# CROP FACE
# ============================================================

def crop_face(image, face_box):
    """
    Crop the detected face from the image.
    """

    x, y, width, height = face_box

    image_height, image_width = image.shape[:2]

    x1 = max(
        0,
        x
    )

    y1 = max(
        0,
        y
    )

    x2 = min(
        image_width,
        x + width
    )

    y2 = min(
        image_height,
        y + height
    )

    if x1 >= x2 or y1 >= y2:
        return None

    face = image[
        y1:y2,
        x1:x2
    ]

    if face.size == 0:
        return None

    return face


# ============================================================
# PROCESS ONE CLASS
# ============================================================

def process_class(
    input_directory,
    output_directory,
    detector,
    class_name
):

    image_files = sorted(
        [
            file
            for file in input_directory.iterdir()
            if (
                file.is_file()
                and file.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]
    )

    total_images = len(image_files)

    processed_count = 0
    skipped_count = 0

    print()
    print("=" * 60)
    print(
        f"Processing class: "
        f"{class_name.upper()}"
    )
    print(
        f"Input : {input_directory}"
    )
    print(
        f"Output: {output_directory}"
    )
    print(
        f"Images: {total_images}"
    )
    print("=" * 60)

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = load_image(
            image_path
        )

        if image is None:

            skipped_count += 1

            print(
                f"[SKIP] Could not read: "
                f"{image_path.name}"
            )

            continue

        # ----------------------------------------------------
        # DETECT LARGEST FACE
        # ----------------------------------------------------

        face_box = detect_largest_face(
            image,
            detector
        )

        if face_box is None:

            skipped_count += 1

            print(
                f"[SKIP] No face detected: "
                f"{image_path.name}"
            )

            continue

        # ----------------------------------------------------
        # CROP FACE
        # ----------------------------------------------------

        face = crop_face(
            image,
            face_box
        )

        if face is None:

            skipped_count += 1

            print(
                f"[SKIP] Invalid face crop: "
                f"{image_path.name}"
            )

            continue

        # ----------------------------------------------------
        # SAVE FACE
        # ----------------------------------------------------

        output_path = (
            output_directory
            / f"{image_path.stem}.jpg"
        )

        success = save_image(
            face,
            output_path
        )

        if not success:

            skipped_count += 1

            print(
                f"[SKIP] Could not save: "
                f"{image_path.name}"
            )

            continue

        processed_count += 1

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            index % 100 == 0
            or index == total_images
        ):

            print(
                f"[{index}/{total_images}] "
                f"Processed: {processed_count} | "
                f"Skipped: {skipped_count}"
            )

    return (
        processed_count,
        skipped_count
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        "MEDIA FORENSICS DATASET PREPROCESSING"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # VERIFY RAW DATASET
    # --------------------------------------------------------

    if not REAL_RAW.exists():

        print()
        print(
            "ERROR: Real dataset directory "
            "not found:"
        )
        print(REAL_RAW)

        sys.exit(1)

    if not FAKE_RAW.exists():

        print()
        print(
            "ERROR: Fake dataset directory "
            "not found:"
        )
        print(FAKE_RAW)

        sys.exit(1)

    # --------------------------------------------------------
    # LOAD FACE DETECTOR
    # --------------------------------------------------------

    print()
    print(
        "Loading face detector..."
    )

    try:

        detector = create_face_detector()

    except Exception as error:

        print()
        print(
            "ERROR: Could not load "
            "face detector."
        )
        print(error)

        sys.exit(1)

    print(
        "Face detector loaded successfully."
    )

    # --------------------------------------------------------
    # PROCESS REAL
    # --------------------------------------------------------

    real_processed, real_skipped = process_class(
        REAL_RAW,
        REAL_PROCESSED,
        detector,
        "real"
    )

    # --------------------------------------------------------
    # PROCESS FAKE
    # --------------------------------------------------------

    fake_processed, fake_skipped = process_class(
        FAKE_RAW,
        FAKE_PROCESSED,
        detector,
        "fake"
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    total_processed = (
        real_processed
        + fake_processed
    )

    total_skipped = (
        real_skipped
        + fake_skipped
    )

    print()
    print("=" * 60)
    print(
        "PREPROCESSING COMPLETE"
    )
    print("=" * 60)

    print()
    print("REAL")

    print(
        f"  Processed: "
        f"{real_processed}"
    )

    print(
        f"  Skipped  : "
        f"{real_skipped}"
    )

    print()
    print("FAKE")

    print(
        f"  Processed: "
        f"{fake_processed}"
    )

    print(
        f"  Skipped  : "
        f"{fake_skipped}"
    )

    print()
    print("TOTAL")

    print(
        f"  Processed: "
        f"{total_processed}"
    )

    print(
        f"  Skipped  : "
        f"{total_skipped}"
    )

    print()
    print(
        "Processed dataset:"
    )

    print(
        PROCESSED_DATASET
    )

    print()
    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()