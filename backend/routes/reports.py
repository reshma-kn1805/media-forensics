from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse


router = APIRouter(
    prefix="/reports",
    tags=["Reports"],
)


REPORTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "reports"
)


@router.get("/{filename}")
def download_report(filename: str):
    """
    Download a generated VERITAS forensic PDF report.
    """

    safe_filename = Path(filename).name

    report_path = REPORTS_DIR / safe_filename

    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Forensic report not found.",
        )

    if report_path.suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF reports can be downloaded.",
        )

    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        filename=safe_filename,
    )