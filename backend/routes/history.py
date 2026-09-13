from fastapi import APIRouter, HTTPException
from sqlalchemy import desc

from backend.database import SessionLocal
from backend.models.analysis import Analysis


router = APIRouter(
    prefix="/history",
    tags=["History"],
)


@router.get("")
def get_analysis_history():
    """
    Return saved media analysis records from the database.
    """

    db = SessionLocal()

    try:
        records = (
            db.query(Analysis)
            .order_by(desc(Analysis.created_at))
            .all()
        )

        history = []

        for record in records:
            history.append(
                {
                    "id": record.id,
                    "filename": record.filename,
                    "content_type": record.content_type,
                    "image_width": record.image_width,
                    "image_height": record.image_height,
                    "faces_detected": record.faces_detected,
                    "prediction": record.prediction,
                    "confidence": record.confidence,
                    "real_probability": record.real_probability,
                    "fake_probability": record.fake_probability,
                    "model_name": record.model_name,
                    "model_version": record.model_version,
                    "explainability_method": record.explainability_method,
                    "explainability_status": record.explainability_status,
                    "processing_time_ms": record.processing_time_ms,
                    "notes": record.notes,
                    "created_at": (
                        record.created_at.isoformat()
                        if record.created_at
                        else None
                    ),
                }
            )

        return {
            "count": len(history),
            "records": history,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to retrieve analysis history: {exc}",
        )

    finally:
        db.close()