from pathlib import Path
import shutil
import tempfile

import cv2


class FaceDetector:
    """
    OpenCV Haar Cascade based face detector.

    Detects faces in an image and returns bounding boxes
    in the format:

        (x, y, width, height)
    """

    def __init__(self):
        # ----------------------------------------------------
        # Locate the Haar Cascade using OpenCV's installation
        # instead of depending on the OneDrive Unicode path.
        # ----------------------------------------------------

        cascade_source = Path(cv2.data.haarcascades) / (
            "haarcascade_frontalface_default.xml"
        )

        if not cascade_source.exists():
            raise FileNotFoundError(
                "OpenCV Haar Cascade file was not found:\n"
                f"{cascade_source}"
            )

        # ----------------------------------------------------
        # Copy the cascade to a simple temporary path.
        # This avoids Windows Unicode/OneDrive path problems.
        # ----------------------------------------------------

        temp_dir = Path(tempfile.gettempdir())

        temp_cascade_path = (
            temp_dir / "media_forensics_haarcascade.xml"
        )

        shutil.copy2(
            cascade_source,
            temp_cascade_path
        )

        # ----------------------------------------------------
        # Load the detector.
        # ----------------------------------------------------

        self.detector = cv2.CascadeClassifier(
            str(temp_cascade_path)
        )

        if self.detector.empty():
            raise RuntimeError(
                "Failed to load the Haar Cascade face detector."
            )

    def detect_faces(self, image):
        """
        Detect faces in a BGR OpenCV image.

        Returns:
            List of bounding boxes:

            [
                (x, y, width, height),
                ...
            ]
        """

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        return list(faces)


def create_face_detector():
    """
    Create and return a FaceDetector instance.
    """

    return FaceDetector()


if __name__ == "__main__":
    detector = create_face_detector()

    print("Face detector created successfully!")
    print(
        "Detector loaded:",
        not detector.detector.empty()
    )