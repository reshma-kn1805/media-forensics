from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)

from backend.database import Base


class Analysis(Base):
    """
    Database model for storing media analysis results.
    """

    __tablename__ = "analyses"

    # ========================================================
    # PRIMARY KEY
    # ========================================================

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    # ========================================================
    # FILE INFORMATION
    # ========================================================

    filename = Column(
        String(255),
        nullable=False
    )

    content_type = Column(
        String(100),
        nullable=False
    )

    # ========================================================
    # IMAGE INFORMATION
    # ========================================================

    image_width = Column(
        Integer,
        nullable=True
    )

    image_height = Column(
        Integer,
        nullable=True
    )

    # ========================================================
    # FACE DETECTION
    # ========================================================

    faces_detected = Column(
        Integer,
        nullable=True
    )

    # ========================================================
    # MODEL RESULT
    # ========================================================

    prediction = Column(
        String(20),
        nullable=True
    )

    confidence = Column(
        Float,
        nullable=True
    )

    real_probability = Column(
        Float,
        nullable=True
    )

    fake_probability = Column(
        Float,
        nullable=True
    )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    model_name = Column(
        String(100),
        nullable=True
    )

    model_version = Column(
        String(50),
        nullable=True
    )

    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    explainability_method = Column(
        String(100),
        nullable=True
    )

    explainability_status = Column(
        String(100),
        nullable=True
    )

    # ========================================================
    # PROCESSING INFORMATION
    # ========================================================

    processing_time_ms = Column(
        Float,
        nullable=True
    )

    # ========================================================
    # OPTIONAL NOTES
    # ========================================================

    notes = Column(
        Text,
        nullable=True
    )

    # ========================================================
    # TIMESTAMP
    # ========================================================

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    def __repr__(self):
        return (
            f"<Analysis("
            f"id={self.id}, "
            f"filename='{self.filename}', "
            f"prediction='{self.prediction}'"
            f")>"
        )
# ============================================================
# PASSKEY / WEBAUTHN CREDENTIAL
# ============================================================

class PasskeyCredential(Base):
    """
    Database model for storing a registered WebAuthn passkey.

    The private key remains securely stored on the user's
    authenticator/device.

    The server stores only the credential ID, public key,
    and authentication metadata required for verification.
    """

    __tablename__ = "passkey_credentials"

    # ========================================================
    # PRIMARY KEY
    # ========================================================

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    # ========================================================
    # USER INFORMATION
    # ========================================================

    username = Column(
        String(100),
        nullable=False,
        index=True
    )

    # ========================================================
    # WEBAUTHN CREDENTIAL
    # ========================================================

    credential_id = Column(
        String(500),
        nullable=False,
        unique=True,
        index=True
    )

    credential_public_key = Column(
        Text,
        nullable=False
    )

    # ========================================================
    # AUTHENTICATOR INFORMATION
    # ========================================================

    sign_count = Column(
        Integer,
        nullable=False,
        default=0
    )

    aaguid = Column(
        String(100),
        nullable=True
    )

    credential_type = Column(
        String(100),
        nullable=True
    )

    device_type = Column(
        String(100),
        nullable=True
    )

    credential_backed_up = Column(
        Integer,
        nullable=False,
        default=0
    )

    user_verified = Column(
        Integer,
        nullable=False,
        default=0
    )

    # ========================================================
    # TIMESTAMP
    # ========================================================

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    last_used_at = Column(
        DateTime,
        nullable=True
    )

    def __repr__(self):
        return (
            f"<PasskeyCredential("
            f"id={self.id}, "
            f"username='{self.username}', "
            f"credential_id='{self.credential_id[:16]}...'"
            f")>"
        )