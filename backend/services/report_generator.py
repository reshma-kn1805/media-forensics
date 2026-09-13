from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

from PIL import Image

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as ReportImage,
)


BASE_DIR = Path(__file__).resolve().parents[2]

REPORT_DIR = BASE_DIR / "backend" / "data" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def safe_value(value, default="Not available"):
    if value is None:
        return default

    if isinstance(value, str) and not value.strip():
        return default

    return value


def percentage(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "Not available"


def number(value, digits=2):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "Not available"


def create_table(rows, widths):
    table = Table(
        rows,
        colWidths=widths,
        repeatRows=1,
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#171b34"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "TEXTCOLOR",
                    (0, 1),
                    (-1, -1),
                    colors.HexColor("#202636"),
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (-1, -1),
                    "Helvetica",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8.5,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#c8ccd8"),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    return table


def create_heatmap_image(heatmap_base64):
    if not heatmap_base64:
        return None

    try:
        image_data = base64.b64decode(heatmap_base64)

        image = Image.open(
            io.BytesIO(image_data)
        ).convert("RGB")

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="PNG",
        )

        buffer.seek(0)

        return ReportImage(
            buffer,
            width=150 * mm,
            height=105 * mm,
        )

    except Exception:
        return None


def generate_forensic_report(
    analysis_result: dict,
    image_bytes: bytes,
):
    analysis_id = analysis_result.get(
        "analysis_id"
    )

    filename = safe_value(
        analysis_result.get("filename"),
        "unknown",
    )

    content_type = safe_value(
        analysis_result.get("content_type"),
        "unknown",
    )

    model = analysis_result.get(
        "model",
        {},
    )

    image = analysis_result.get(
        "image",
        {},
    )

    face_detection = analysis_result.get(
        "face_detection",
        {},
    )

    metadata = analysis_result.get(
        "metadata",
        {},
    )

    integrity = analysis_result.get(
        "image_integrity",
        {},
    )

    compression = analysis_result.get(
        "compression_analysis",
        {},
    )

    noise_frequency = analysis_result.get(
        "noise_frequency_analysis",
        {},
    )

    explainability = analysis_result.get(
        "explainability",
        {},
    )

    processing_time = analysis_result.get(
        "processing_time_ms"
    )

    report_filename = (
        f"VERITAS_Report_{analysis_id}.pdf"
    )

    report_path = REPORT_DIR / report_filename

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="VeritasTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#202640"),
            spaceAfter=6,
        )
    )

    styles.add(
        ParagraphStyle(
            name="VeritasSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#667085"),
            spaceAfter=18,
        )
    )

    styles.add(
        ParagraphStyle(
            name="VeritasSection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#252b45"),
            spaceBefore=12,
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="VeritasBody",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=13,
            textColor=colors.HexColor("#4b5565"),
        )
    )

    styles.add(
        ParagraphStyle(
            name="VeritasVerdict",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            alignment=TA_CENTER,
            spaceBefore=8,
            spaceAfter=10,
        )
    )

    document = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="VERITAS Forensic Report",
        author="VERITAS Media Forensics",
    )

    story = []

    # ========================================================
    # HEADER
    # ========================================================

    story.append(
        Paragraph(
            "VERITAS",
            styles["VeritasTitle"],
        )
    )

    story.append(
        Paragraph(
            "MEDIA FORENSICS",
            styles["VeritasSubtitle"],
        )
    )

    story.append(
        Paragraph(
            "Forensic Investigation Report",
            styles["Heading1"],
        )
    )

    story.append(Spacer(1, 8))

    # ========================================================
    # CASE INFORMATION
    # ========================================================

    story.append(
        Paragraph(
            "CASE INFORMATION",
            styles["VeritasSection"],
        )
    )

    case_rows = [
        ["Field", "Value"],
        [
            "Analysis ID",
            str(
                safe_value(
                    analysis_id
                )
            ),
        ],
        [
            "Submitted File",
            str(filename),
        ],
        [
            "Content Type",
            str(content_type),
        ],
        [
            "Analysis Date",
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        ],
        [
            "Processing Time",
            f"{number(processing_time)} ms",
        ],
    ]

    story.append(
        create_table(
            case_rows,
            [55 * mm, 115 * mm],
        )
    )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    story.append(
        Paragraph(
            "AI CLASSIFICATION",
            styles["VeritasSection"],
        )
    )

    prediction = str(
        safe_value(
            model.get("prediction"),
            "UNKNOWN",
        )
    ).upper()

    if prediction == "FAKE":
        verdict_color = colors.HexColor(
            "#b42318"
        )
    elif prediction == "REAL":
        verdict_color = colors.HexColor(
            "#087443"
        )
    else:
        verdict_color = colors.HexColor(
            "#667085"
        )

    verdict_style = ParagraphStyle(
        name="DynamicVerdict",
        parent=styles["VeritasVerdict"],
        textColor=verdict_color,
    )

    story.append(
        Paragraph(
            prediction,
            verdict_style,
        )
    )

    classification_rows = [
        ["Metric", "Result"],
        [
            "Model",
            str(
                safe_value(
                    model.get("name")
                )
            ),
        ],
        [
            "Prediction",
            prediction,
        ],
        [
            "Model Confidence",
            percentage(
                model.get("confidence")
            ),
        ],
        [
            "Real Probability",
            percentage(
                model.get(
                    "real_probability"
                )
            ),
        ],
        [
            "Fake Probability",
            percentage(
                model.get(
                    "fake_probability"
                )
            ),
        ],
        [
            "Faces Detected",
            str(
                safe_value(
                    face_detection.get(
                        "faces_detected"
                    ),
                    0,
                )
            ),
        ],
        [
            "Image Dimensions",
            (
                f"{safe_value(image.get('width'))} "
                f"× "
                f"{safe_value(image.get('height'))}"
            ),
        ],
    ]

    story.append(
        create_table(
            classification_rows,
            [65 * mm, 105 * mm],
        )
    )

    # ========================================================
    # FACE DETECTION
    # ========================================================

    story.append(
        Paragraph(
            "FACE DETECTION",
            styles["VeritasSection"],
        )
    )

    selected_face = face_detection.get(
        "selected_face",
        {},
    )

    face_rows = [
        ["Property", "Value"],
        [
            "Detection Method",
            str(
                safe_value(
                    face_detection.get(
                        "method"
                    ),
                    "OpenCV Haar Cascade",
                )
            ),
        ],
        [
            "Faces Detected",
            str(
                safe_value(
                    face_detection.get(
                        "faces_detected"
                    ),
                    0,
                )
            ),
        ],
        [
            "Selected Face X",
            str(
                safe_value(
                    selected_face.get("x")
                )
            ),
        ],
        [
            "Selected Face Y",
            str(
                safe_value(
                    selected_face.get("y")
                )
            ),
        ],
        [
            "Selected Face Width",
            str(
                safe_value(
                    selected_face.get("width")
                )
            ),
        ],
        [
            "Selected Face Height",
            str(
                safe_value(
                    selected_face.get("height")
                )
            ),
        ],
    ]

    story.append(
        create_table(
            face_rows,
            [65 * mm, 105 * mm],
        )
    )

    # ========================================================
    # GRAD-CAM
    # ========================================================

    story.append(
        Paragraph(
            "EXPLAINABILITY",
            styles["VeritasSection"],
        )
    )

    explanation_rows = [
        ["Property", "Value"],
        [
            "Method",
            str(
                safe_value(
                    explainability.get(
                        "method"
                    ),
                    "Grad-CAM",
                )
            ),
        ],
        [
            "Status",
            str(
                safe_value(
                    explainability.get(
                        "status"
                    )
                )
            ),
        ],
        [
            "Target Class",
            prediction,
        ],
    ]

    story.append(
        create_table(
            explanation_rows,
            [65 * mm, 105 * mm],
        )
    )

    story.append(Spacer(1, 8))

    heatmap = create_heatmap_image(
        explainability.get(
            "heatmap_base64"
        )
    )

    if heatmap is not None:

        story.append(
            Paragraph(
                "Grad-CAM Explanation Heatmap",
                styles["Heading3"],
            )
        )

        story.append(
            heatmap
        )

        story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "Grad-CAM is an interpretability aid "
            "showing image regions that influenced "
            "the model output. It is not independent "
            "proof of manipulation or authenticity.",
            styles["VeritasBody"],
        )
    )

    # ========================================================
    # METADATA
    # ========================================================

    story.append(
        Paragraph(
            "METADATA / EXIF ANALYSIS",
            styles["VeritasSection"],
        )
    )

    exif_status = metadata.get(
        "exif_status"
    )

    if exif_status is None:
        exif_status = (
            "PRESENT"
            if metadata.get(
                "exif_present"
            )
            else "NOT PRESENT"
        )

    gps_status = metadata.get(
        "gps_status"
    )

    if gps_status is None:
        gps_status = safe_value(
            metadata.get("gps"),
            "NOT PRESENT",
        )

    metadata_rows = [
        ["Property", "Value"],
        [
            "Image Format",
            str(
                safe_value(
                    metadata.get(
                        "format"
                    )
                )
            ),
        ],
        [
            "EXIF Status",
            str(exif_status),
        ],
        [
            "Camera Make",
            str(
                safe_value(
                    metadata.get(
                        "camera_make"
                    )
                )
            ),
        ],
        [
            "Camera Model",
            str(
                safe_value(
                    metadata.get(
                        "camera_model"
                    )
                )
            ),
        ],
        [
            "Date / Time",
            str(
                safe_value(
                    metadata.get(
                        "date_time",
                        metadata.get(
                            "date_taken"
                        ),
                    )
                )
            ),
        ],
        [
            "Software",
            str(
                safe_value(
                    metadata.get(
                        "software"
                    )
                )
            ),
        ],
        [
            "Orientation",
            str(
                safe_value(
                    metadata.get(
                        "orientation"
                    )
                )
            ),
        ],
        [
            "GPS",
            str(gps_status),
        ],
    ]

    story.append(
        create_table(
            metadata_rows,
            [65 * mm, 105 * mm],
        )
    )

    # ========================================================
    # IMAGE INTEGRITY
    # ========================================================

    story.append(
        Paragraph(
            "IMAGE INTEGRITY",
            styles["VeritasSection"],
        )
    )

    file_size_kb = integrity.get(
        "file_size_kb"
    )

    if file_size_kb is None:
        file_size_bytes = integrity.get(
            "file_size_bytes"
        )

        if file_size_bytes is not None:
            file_size_kb = (
                float(file_size_bytes)
                / 1024
            )

    integrity_rows = [
        ["Property", "Value"],
        [
            "Status",
            str(
                safe_value(
                    integrity.get(
                        "status"
                    )
                )
            ),
        ],
        [
            "Format",
            str(
                safe_value(
                    integrity.get(
                        "format"
                    )
                )
            ),
        ],
        [
            "File Size",
            (
                f"{number(file_size_kb)} KB"
                if file_size_kb is not None
                else "Not available"
            ),
        ],
        [
            "SHA-256",
            str(
                safe_value(
                    integrity.get(
                        "sha256"
                    )
                )
            ),
        ],
        [
            "Verification",
            str(
                safe_value(
                    integrity.get(
                        "verification"
                    )
                )
            ),
        ],
        [
            "Decode Check",
            str(
                safe_value(
                    integrity.get(
                        "decode_check"
                    )
                )
            ),
        ],
    ]

    story.append(
        create_table(
            integrity_rows,
            [65 * mm, 105 * mm],
        )
    )

    # ========================================================
    # COMPRESSION
    # ========================================================

    story.append(
        Paragraph(
            "COMPRESSION ANALYSIS",
            styles["VeritasSection"],
        )
    )

    compression_level = compression.get(
        "compression_level"
    )

    if compression_level is None:
        compression_level = compression.get(
            "estimated_compression_level"
        )

    compression_rows = [
        ["Property", "Value"],
        [
            "Status",
            str(
                safe_value(
                    compression.get(
                        "status"
                    )
                )
            ),
        ],
        [
            "Format",
            str(
                safe_value(
                    compression.get(
                        "format"
                    )
                )
            ),
        ],
        [
            "Compression Level",
            str(
                safe_value(
                    compression_level
                )
            ),
        ],
        [
            "Bytes / Pixel",
            str(
                safe_value(
                    compression.get(
                        "bytes_per_pixel"
                    )
                )
            ),
        ],
        [
            "Pixel Count",
            str(
                safe_value(
                    compression.get(
                        "pixel_count"
                    )
                )
            ),
        ],
        [
            "Decode Check",
            str(
                safe_value(
                    compression.get(
                        "decode_check"
                    )
                )
            ),
        ],
    ]

    story.append(
        create_table(
            compression_rows,
            [65 * mm, 105 * mm],
        )
    )

    # ========================================================
    # NOISE / FREQUENCY
    # ========================================================

    story.append(
        Paragraph(
            "NOISE & FREQUENCY ANALYSIS",
            styles["VeritasSection"],
        )
    )

    high_frequency_ratio = (
        noise_frequency.get(
            "high_frequency_ratio"
        )
    )

    if high_frequency_ratio is None:
        high_frequency_ratio = (
            noise_frequency.get(
                "high_frequency_energy_ratio"
            )
        )

    if high_frequency_ratio is not None:
        try:
            ratio_value = float(
                high_frequency_ratio
            )

            if ratio_value <= 1:
                high_frequency_display = (
                    f"{ratio_value * 100:.2f}%"
                )
            else:
                high_frequency_display = (
                    f"{ratio_value:.2f}%"
                )

        except (TypeError, ValueError):
            high_frequency_display = (
                "Not available"
            )
    else:
        high_frequency_display = (
            "Not available"
        )

    sharpness = noise_frequency.get(
        "sharpness"
    )

    if sharpness is None:
        sharpness = noise_frequency.get(
            "sharpness_variance"
        )

    frequency_profile = (
        noise_frequency.get(
            "frequency_profile"
        )
    )

    if frequency_profile is None:
        frequency_profile = (
            noise_frequency.get(
                "frequency_characteristic"
            )
        )

    noise_rows = [
        ["Property", "Value"],
        [
            "Status",
            str(
                safe_value(
                    noise_frequency.get(
                        "status"
                    )
                )
            ),
        ],
        [
            "Noise Level",
            str(
                safe_value(
                    noise_frequency.get(
                        "noise_level"
                    )
                )
            ),
        ],
        [
            "Noise Variance",
            str(
                safe_value(
                    noise_frequency.get(
                        "noise_variance"
                    )
                )
            ),
        ],
        [
            "Sharpness",
            str(
                safe_value(
                    sharpness
                )
            ),
        ],
        [
            "High-Frequency Ratio",
            high_frequency_display,
        ],
        [
            "Frequency Profile",
            str(
                safe_value(
                    frequency_profile
                )
            ),
        ],
    ]

    story.append(
        create_table(
            noise_rows,
            [65 * mm, 105 * mm],
        )
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    story.append(
        Paragraph(
            "FORENSIC INTERPRETATION",
            styles["VeritasSection"],
        )
    )

    story.append(
        Paragraph(
            (
                "The classification result represents "
                "the output of the trained EfficientNet-B0 "
                "model for the detected facial region. "
                "Metadata, integrity, compression, noise, "
                "and frequency measurements are supporting "
                "forensic indicators. These indicators "
                "should be interpreted together and do not "
                "individually establish whether an image "
                "has been manipulated."
            ),
            styles["VeritasBody"],
        )
    )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    story.append(
        Paragraph(
            "VERITAS FORENSIC DISCLAIMER",
            styles["VeritasSection"],
        )
    )

    story.append(
        Paragraph(
            (
                "VERITAS provides automated analytical "
                "findings intended to support media-forensic "
                "investigation. AI classification, "
                "Grad-CAM visualizations, metadata observations, "
                "compression measurements, and noise/frequency "
                "signals are not presented as conclusive proof "
                "of authenticity or manipulation. Final "
                "determinations should consider the complete "
                "evidence context and, where appropriate, "
                "qualified human forensic examination."
            ),
            styles["VeritasBody"],
        )
    )

    # ========================================================
    # FOOTER
    # ========================================================

    def add_page_number(canvas, doc):
        canvas.saveState()

        canvas.setFont(
            "Helvetica",
            7,
        )

        canvas.setFillColor(
            colors.HexColor("#667085")
        )

        canvas.drawCentredString(
            A4[0] / 2,
            10 * mm,
            (
                f"VERITAS Media Forensics  •  "
                f"Page {doc.page}"
            ),
        )

        canvas.restoreState()

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    return str(report_path)