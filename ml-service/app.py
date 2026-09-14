from pathlib import Path
import io
import time

import cv2
import numpy as np
import torch

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ExifTags

from model import create_model
from utils.face_detector import create_face_detector
from utils.preprocessing import preprocess_image
from utils.gradcam import generate_gradcam_base64

from backend.services.forensics import run_forensics
from backend.services.report_generator import generate_forensic_report

from backend.database import SessionLocal, create_database
from backend.models.analysis import Analysis

from backend.routes.auth import router as auth_router
from backend.routes.history import router as history_router
from backend.routes.reports import router as reports_router


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="VERITAS Media Forensics API",
    description="AI-powered media forensic analysis service.",
    version="1.0.0",
)


# ============================================================
# DATABASE STARTUP
# ============================================================

@app.on_event("startup")
def initialize_database():
    """
    Create all required database tables when the
    VERITAS backend starts.
    """
    create_database()
    print("VERITAS database initialized successfully.")


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "https://veritas-frontend-0xoi.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTERS
# ============================================================

app.include_router(auth_router)
app.include_router(history_router)
app.include_router(reports_router)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pt"

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

GENERATE_GRADCAM = True


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = create_model()

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

elif "state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["state_dict"]
    )

else:
    raise RuntimeError(
        "The checkpoint does not contain "
        "'model_state_dict' or 'state_dict'."
    )

model.to(device)
model.eval()


# ============================================================
# MODEL INFORMATION
# ============================================================

class_names = checkpoint.get(
    "class_names",
    ["fake", "real"],
)

if not isinstance(class_names, list):
    class_names = list(class_names)

if len(class_names) != 2:
    raise RuntimeError(
        f"Expected exactly two classes, got: {class_names}"
    )

class_names = [
    str(name).lower()
    for name in class_names
]

if "fake" not in class_names or "real" not in class_names:
    raise RuntimeError(
        "Checkpoint class mapping must contain "
        f"'fake' and 'real'. Found: {class_names}"
    )

trained_epoch = checkpoint.get("epoch")

validation_accuracy = checkpoint.get(
    "validation_accuracy"
)

trained_for_deepfake_detection = True


print("=" * 60)
print("VERITAS ML SERVICE")
print("=" * 60)
print(f"Model: {MODEL_PATH}")
print(f"Device: {device}")
print(f"Class names: {class_names}")
print(f"Training epoch: {trained_epoch}")
print(
    f"Validation accuracy: "
    f"{validation_accuracy}"
)
print(
    f"Grad-CAM enabled: "
    f"{GENERATE_GRADCAM}"
)
print("=" * 60)


# ============================================================
# FACE DETECTOR
# ============================================================

face_detector = create_face_detector()


# ============================================================
# IMAGE DECODING
# ============================================================

def decode_image(
    image_bytes: bytes,
) -> np.ndarray:

    array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            "The uploaded file could not be decoded "
            "as a valid image."
        )

    return image


# ============================================================
# FACE AREA
# ============================================================

def get_face_area(face):
    """
    Calculate the area of a detected face.

    Expected face format:
        [x, y, width, height]
    """

    return (
        max(0, int(face[2]))
        * max(0, int(face[3]))
    )


# ============================================================
# LARGEST FACE
# ============================================================

def get_largest_face(faces):

    if not faces:
        raise ValueError(
            "No faces were detected."
        )

    return max(
        faces,
        key=get_face_area,
    )


# ============================================================
# FACE BOX NORMALIZATION
# ============================================================

def normalize_face_box(
    face,
    image_width,
    image_height,
):
    """
    Convert a detected face into a safe bounding box.

    The returned values are clipped to the image boundaries.
    """

    x = int(face[0])
    y = int(face[1])
    width = int(face[2])
    height = int(face[3])

    x = max(
        0,
        min(
            x,
            image_width - 1,
        ),
    )

    y = max(
        0,
        min(
            y,
            image_height - 1,
        ),
    )

    right = max(
        x + 1,
        min(
            x + width,
            image_width,
        ),
    )

    bottom = max(
        y + 1,
        min(
            y + height,
            image_height,
        ),
    )

    normalized_width = right - x
    normalized_height = bottom - y

    return (
        x,
        y,
        normalized_width,
        normalized_height,
    )


# ============================================================
# LEGACY METADATA FORMATTER
# ============================================================

def extract_metadata(
    image_bytes: bytes,
) -> dict:

    metadata = {
        "format": "Unknown",
        "exif_available": False,
        "camera_make": None,
        "camera_model": None,
        "date_taken": None,
        "software": None,
        "orientation": None,
        "gps": None,
    }

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        metadata["format"] = (
            image.format or "Unknown"
        )

        exif = image.getexif()

        if not exif:
            return metadata

        metadata["exif_available"] = True

        for tag_id, value in exif.items():

            tag_name = ExifTags.TAGS.get(
                tag_id,
                str(tag_id),
            )

            if tag_name == "Make":
                metadata["camera_make"] = str(value)

            elif tag_name == "Model":
                metadata["camera_model"] = str(value)

            elif tag_name in {
                "DateTime",
                "DateTimeOriginal",
                "DateTimeDigitized",
            }:

                if metadata["date_taken"] is None:
                    metadata["date_taken"] = str(value)

            elif tag_name == "Software":
                metadata["software"] = str(value)

            elif tag_name == "Orientation":
                metadata["orientation"] = value

            elif tag_name == "GPSInfo":
                metadata["gps"] = "Present"

    except Exception as error:

        print(
            "Metadata analysis warning:",
            error,
        )

    return metadata


# ============================================================
# JSON SAFE CONVERSION
# ============================================================

def make_json_safe(value):

    if isinstance(value, dict):

        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):

        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        (
            np.integer,
            np.int64,
            np.int32,
        ),
    ):

        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            np.float64,
            np.float32,
        ),
    ):

        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


# ============================================================
# MODEL PREDICTION
# ============================================================

def calculate_prediction(
    input_tensor,
):

    model.eval()

    with torch.inference_mode():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1,
        )[0]

    fake_index = class_names.index("fake")
    real_index = class_names.index("real")

    fake_probability = float(
        probabilities[fake_index].item()
    )

    real_probability = float(
        probabilities[real_index].item()
    )

    predicted_index = int(
        torch.argmax(
            probabilities
        ).item()
    )

    prediction = class_names[
        predicted_index
    ]

    confidence = float(
        probabilities[
            predicted_index
        ].item()
    )

    return (
        prediction,
        confidence,
        real_probability,
        fake_probability,
    )


# ============================================================
# ANALYZE SINGLE FACE
# ============================================================

def analyze_single_face(
    image: np.ndarray,
    face,
    face_index: int,
):
    """
    Run EfficientNet prediction for one detected face.

    IMPORTANT PERFORMANCE OPTIMIZATION:
    Grad-CAM is NOT generated here.

    All detected faces receive normal model prediction.
    Grad-CAM is generated only once later for the
    selected/highest-risk face.
    """

    image_height, image_width = (
        image.shape[:2]
    )

    (
        x,
        y,
        width,
        height,
    ) = normalize_face_box(
        face=face,
        image_width=image_width,
        image_height=image_height,
    )

    face_crop = image[
        y:y + height,
        x:x + width,
    ]

    if face_crop.size == 0:

        raise ValueError(
            f"Face {face_index} could not be cropped."
        )

    face_rgb = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2RGB,
    )

    face_pil = Image.fromarray(
        face_rgb
    )

    input_tensor = preprocess_image(
        face_pil
    )

    input_tensor = input_tensor.to(
        device
    )

    (
        prediction,
        confidence,
        real_probability,
        fake_probability,
    ) = calculate_prediction(
        input_tensor
    )

    return {
        "face_index": int(face_index),

        "bounding_box": {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
        },

        "area": int(
            width * height
        ),

        "prediction": prediction,

        "confidence": confidence,

        "real_probability": (
            real_probability
        ),

        "fake_probability": (
            fake_probability
        ),

        "explainability": {
            "method": "Grad-CAM",
            "status": "pending",
            "heatmap_base64": None,
        },
    }


# ============================================================
# GENERATE GRAD-CAM FOR SELECTED FACE ONLY
# ============================================================

def generate_selected_face_gradcam(
    image: np.ndarray,
    selected_face_result: dict,
):
    """
    Generate Grad-CAM only for the selected face.

    This is intentionally performed once per image
    rather than once for every detected face.

    This significantly reduces CPU and memory usage
    on constrained deployment environments such as Render.
    """

    if not GENERATE_GRADCAM:
        selected_face_result[
            "explainability"
        ]["status"] = "disabled"

        return selected_face_result

    bounding_box = selected_face_result[
        "bounding_box"
    ]

    x = int(
        bounding_box["x"]
    )

    y = int(
        bounding_box["y"]
    )

    width = int(
        bounding_box["width"]
    )

    height = int(
        bounding_box["height"]
    )

    face_crop = image[
        y:y + height,
        x:x + width,
    ]

    if face_crop.size == 0:

        selected_face_result[
            "explainability"
        ]["status"] = "error"

        return selected_face_result

    face_rgb = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2RGB,
    )

    face_pil = Image.fromarray(
        face_rgb
    )

    input_tensor = preprocess_image(
        face_pil
    )

    input_tensor = input_tensor.to(
        device
    )

    prediction = selected_face_result[
        "prediction"
    ]

    target_class = class_names.index(
        prediction
    )

    try:

        heatmap_base64 = (
            generate_gradcam_base64(
                model=model,
                input_tensor=input_tensor,
                face_image=face_pil,
                target_class=target_class,
            )
        )

        selected_face_result[
            "explainability"
        ] = {
            "method": "Grad-CAM",
            "status": "generated",
            "heatmap_base64": heatmap_base64,
        }

    except Exception as error:

        print(
            "Selected-face Grad-CAM warning:",
            error,
        )

        selected_face_result[
            "explainability"
        ] = {
            "method": "Grad-CAM",
            "status": "error",
            "heatmap_base64": None,
        }

    return selected_face_result


# ============================================================
# OVERALL MULTI-FACE VERDICT
# ============================================================

def calculate_overall_face_verdict(
    face_results,
):
    """
    Calculate an overall image-level verdict from
    all detected facial regions.

    The overall verdict is based on the highest
    fake probability among detected faces.

    This is intentionally conservative:
    if any detected face has a strong fake signal,
    the image-level result reflects that face.
    """

    if not face_results:
        raise ValueError(
            "No face analysis results available."
        )

    highest_fake_face = max(
        face_results,
        key=lambda face: float(
            face["fake_probability"]
        ),
    )

    highest_real_face = max(
        face_results,
        key=lambda face: float(
            face["real_probability"]
        ),
    )

    highest_fake_probability = float(
        highest_fake_face[
            "fake_probability"
        ]
    )

    highest_real_probability = float(
        highest_real_face[
            "real_probability"
        ]
    )

    if (
        highest_fake_probability
        >= highest_real_probability
    ):

        overall_prediction = "fake"

    else:

        overall_prediction = "real"

    if overall_prediction == "fake":

        confidence = (
            highest_fake_probability
        )

    else:

        confidence = (
            highest_real_probability
        )

    return (
        overall_prediction,
        float(confidence),
        highest_real_probability,
        highest_fake_probability,
        highest_fake_face,
    )


# ============================================================
# DATABASE SAVE
# ============================================================

def save_analysis_to_database(
    filename,
    content_type,
    image_width,
    image_height,
    faces_detected,
    prediction,
    confidence,
    real_probability,
    fake_probability,
    processing_time_ms,
    explainability_status,
):

    db = SessionLocal()

    try:

        record = Analysis(
            filename=filename,
            content_type=content_type,
            image_width=image_width,
            image_height=image_height,
            faces_detected=faces_detected,
            prediction=prediction,
            confidence=confidence,
            real_probability=real_probability,
            fake_probability=fake_probability,
            model_name="EfficientNet-B0",
            model_version="1.0",
            explainability_method="Grad-CAM",
            explainability_status=(
                explainability_status
            ),
            processing_time_ms=(
                processing_time_ms
            ),
            notes=(
                "Analysis performed using the "
                "trained VERITAS EfficientNet-B0 "
                "model with multi-face analysis "
                "and supporting forensic indicators. "
                "Grad-CAM generated only for the "
                "selected highest-risk face."
            ),
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return record.id

    finally:
        db.close()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": (
            "VERITAS Media Forensics API"
        ),
        "status": "running",
        "model": "EfficientNet-B0",
        "trained_for_deepfake_detection": True,
        "gradcam_enabled": GENERATE_GRADCAM,
        "gradcam_strategy": (
            "Selected face only"
        ),
        "multi_face_analysis": True,
        "forensic_modules": {
            "metadata": True,
            "integrity": True,
            "compression": True,
            "noise": True,
            "frequency": True,
            "reporting": True,
        },
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",

        "model_loaded": True,

        "face_detector_loaded": (
            face_detector is not None
        ),

        "device": str(device),

        "trained_for_deepfake_detection": (
            trained_for_deepfake_detection
        ),

        "trained_epoch": trained_epoch,

        "validation_accuracy": (
            validation_accuracy
        ),

        "class_names": class_names,

        "metadata_analysis_available": True,

        "image_integrity_analysis_available": True,

        "compression_analysis_available": True,

        "noise_frequency_analysis_available": True,

        "multi_face_analysis_available": True,

        "report_generation_available": True,

        "gradcam_enabled": GENERATE_GRADCAM,

        "gradcam_method": "Grad-CAM",

        "gradcam_target_layer": (
            "EfficientNet-B0 features[-1]"
        ),

        "gradcam_strategy": (
            "Selected highest-risk face only"
        ),
    }


# ============================================================
# ANALYZE
# ============================================================

@app.post("/analyze")
async def analyze_media(
    file: UploadFile = File(...),
):

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # 1. VALIDATE FILE
    # --------------------------------------------------------

    if file.content_type not in ALLOWED_CONTENT_TYPES:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Please upload JPEG, PNG, or WebP."
            ),
        )

    # --------------------------------------------------------
    # 2. READ FILE
    # --------------------------------------------------------

    image_bytes = await file.read()

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    # --------------------------------------------------------
    # 3. DECODE IMAGE
    # --------------------------------------------------------

    try:

        image = decode_image(
            image_bytes
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    image_height, image_width = (
        image.shape[:2]
    )

    # --------------------------------------------------------
    # 4. OPEN PIL IMAGE
    # --------------------------------------------------------

    try:

        pil_image = Image.open(
            io.BytesIO(image_bytes)
        )

        pil_image.load()

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded image could not "
                f"be opened: {error}"
            ),
        )

    # --------------------------------------------------------
    # 5. RUN FORENSIC ANALYSIS
    # --------------------------------------------------------

    try:

        forensic_data = run_forensics(
            data=image_bytes,
            image=pil_image,
        )

    except Exception as error:

        print(
            "Forensic analysis warning:",
            error,
        )

        forensic_data = {
            "metadata": extract_metadata(
                image_bytes
            ),

            "integrity": {
                "status": "WARNING",
                "verification": "NOT COMPLETED",
                "sha256": None,
            },

            "compression": {
                "status": "WARNING",
                "compression_level": "Unknown",
            },

            "noise_frequency": {
                "status": "WARNING",
                "noise_level": "Unknown",
                "noise_variance": None,
                "sharpness": None,
                "high_frequency_ratio": None,
                "frequency_profile": "Unknown",
            },
        }

    metadata = forensic_data.get(
        "metadata",
        {},
    )

    integrity = forensic_data.get(
        "integrity",
        {},
    )

    compression = forensic_data.get(
        "compression",
        {},
    )

    noise_frequency = forensic_data.get(
        "noise_frequency",
        {},
    )

    # --------------------------------------------------------
    # 6. FACE DETECTION
    # --------------------------------------------------------

    try:

        faces = face_detector.detect_faces(
            image
        )

    except Exception as error:

        print(
            "Face detection error:",
            error,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Face detection failed: "
                f"{error}"
            ),
        )

    faces_detected = len(faces)

    if faces_detected == 0:

        raise HTTPException(
            status_code=400,
            detail=(
                "No face was detected in the "
                "uploaded image. Please upload "
                "an image containing a clear face."
            ),
        )

    # --------------------------------------------------------
    # 7. ANALYZE EVERY DETECTED FACE
    # --------------------------------------------------------

    face_results = []

    face_analysis_errors = []

    for face_index, face in enumerate(
        faces,
        start=1,
    ):

        try:

            result = analyze_single_face(
                image=image,
                face=face,
                face_index=face_index,
            )

            face_results.append(
                result
            )

        except Exception as error:

            print(
                f"Face {face_index} analysis error:",
                error,
            )

            face_analysis_errors.append(
                {
                    "face_index": int(
                        face_index
                    ),
                    "error": str(error),
                }
            )

    if not face_results:

        raise HTTPException(
            status_code=500,
            detail=(
                "Faces were detected, but none "
                "could be analyzed successfully."
            ),
        )

    # --------------------------------------------------------
    # 8. OVERALL VERDICT
    # --------------------------------------------------------

    (
        prediction,
        confidence,
        real_probability,
        fake_probability,
        selected_face_result,
    ) = calculate_overall_face_verdict(
        face_results
    )

    # --------------------------------------------------------
    # 9. GENERATE GRAD-CAM ONLY FOR SELECTED FACE
    # --------------------------------------------------------

    selected_face_result = (
        generate_selected_face_gradcam(
            image=image,
            selected_face_result=(
                selected_face_result
            ),
        )
    )

    # Update the selected face inside the
    # original face_results list so the frontend
    # receives the generated heatmap there too.

    for index, face_result in enumerate(
        face_results
    ):

        if (
            face_result["face_index"]
            == selected_face_result["face_index"]
        ):

            face_results[index] = (
                selected_face_result
            )

            break

    selected_face = (
        selected_face_result[
            "bounding_box"
        ]
    )

    explainability_status = (
        selected_face_result[
            "explainability"
        ]["status"]
    )

    heatmap_base64 = (
        selected_face_result[
            "explainability"
        ]["heatmap_base64"]
    )

    # --------------------------------------------------------
    # 10. PROCESSING TIME
    # --------------------------------------------------------

    processing_time_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    # --------------------------------------------------------
    # 11. BUILD ANALYSIS RESULT
    # --------------------------------------------------------

    analysis_result = {
        "filename": (
            file.filename
            or "unknown"
        ),

        "content_type": (
            file.content_type
        ),

        "image": {
            "width": image_width,
            "height": image_height,
        },

        "metadata": metadata,

        "image_integrity": integrity,

        "compression_analysis": compression,

        "noise_frequency_analysis": (
            noise_frequency
        ),

        "face_detection": {
            "faces_detected": (
                faces_detected
            ),

            "faces_successfully_analyzed": (
                len(face_results)
            ),

            "method": (
                "OpenCV Haar Cascade"
            ),

            "selected_face": selected_face,

            "faces": face_results,

            "analysis_errors": (
                face_analysis_errors
            ),
        },

        "model": {
            "name": "EfficientNet-B0",

            "prediction": prediction,

            "confidence": confidence,

            "real_probability": (
                real_probability
            ),

            "fake_probability": (
                fake_probability
            ),

            "trained_for_deepfake_detection": (
                trained_for_deepfake_detection
            ),

            "class_names": class_names,

            "trained_epoch": trained_epoch,

            "validation_accuracy": (
                validation_accuracy
            ),
        },

        "explainability": {
            "method": "Grad-CAM",

            "status": explainability_status,

            "heatmap_base64": heatmap_base64,

            "selected_face_index": (
                selected_face_result[
                    "face_index"
                ]
            ),
        },

        "processing_time_ms": (
            processing_time_ms
        ),
    }

    # --------------------------------------------------------
    # 12. SAVE DATABASE RECORD
    # --------------------------------------------------------

    try:

        analysis_id = (
            save_analysis_to_database(
                filename=(
                    file.filename
                    or "unknown"
                ),

                content_type=(
                    file.content_type
                ),

                image_width=image_width,

                image_height=image_height,

                faces_detected=(
                    faces_detected
                ),

                prediction=prediction,

                confidence=confidence,

                real_probability=(
                    real_probability
                ),

                fake_probability=(
                    fake_probability
                ),

                processing_time_ms=(
                    processing_time_ms
                ),

                explainability_status=(
                    explainability_status
                ),
            )
        )

    except Exception as error:

        print(
            "Database error:",
            error,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Analysis completed, but the "
                "result could not be saved "
                f"to the database: {error}"
            ),
        )

    analysis_result["analysis_id"] = (
        analysis_id
    )

    # --------------------------------------------------------
    # 13. GENERATE PDF REPORT
    # --------------------------------------------------------

    report_available = False
    report_path = None
    report_error = None

    try:

        report_path = (
            generate_forensic_report(
                analysis_result=analysis_result,
                image_bytes=image_bytes,
            )
        )

        if report_path:

            report_path = str(
                report_path
            )

            report_available = (
                Path(report_path).exists()
            )

    except Exception as error:

        print(
            "Report generation warning:",
            error,
        )

        report_error = str(error)

    # --------------------------------------------------------
    # 14. FINAL RESPONSE
    # --------------------------------------------------------

    response = {
        "status": "success",

        "message": (
            "Media analysis completed successfully."
        ),

        "analysis_id": analysis_id,

        "filename": (
            file.filename
            or "unknown"
        ),

        "content_type": (
            file.content_type
        ),

        "image": {
            "width": image_width,
            "height": image_height,
        },

        "face_detection": {
            "faces_detected": (
                faces_detected
            ),

            "faces_successfully_analyzed": (
                len(face_results)
            ),

            "method": (
                "OpenCV Haar Cascade"
            ),

            "selected_face": selected_face,

            "faces": face_results,

            "analysis_errors": (
                face_analysis_errors
            ),
        },

        "model": {
            "name": "EfficientNet-B0",

            "prediction": prediction,

            "confidence": confidence,

            "real_probability": (
                real_probability
            ),

            "fake_probability": (
                fake_probability
            ),

            "trained_for_deepfake_detection": (
                trained_for_deepfake_detection
            ),

            "class_names": class_names,

            "trained_epoch": trained_epoch,

            "validation_accuracy": (
                validation_accuracy
            ),
        },

        "explainability": {
            "method": "Grad-CAM",

            "status": explainability_status,

            "heatmap_base64": heatmap_base64,

            "selected_face_index": (
                selected_face_result[
                    "face_index"
                ]
            ),
        },

        "metadata": metadata,

        "image_integrity": integrity,

        "compression_analysis": compression,

        "noise_frequency_analysis": (
            noise_frequency
        ),

        "report": {
            "available": report_available,

            "path": (
                report_path
                if report_available
                else None
            ),

            "error": report_error,
        },

        "processing_time_ms": round(
            processing_time_ms,
            2,
        ),
    }

    return make_json_safe(
        response
    )