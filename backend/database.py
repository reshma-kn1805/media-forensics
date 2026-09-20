from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_DIR = PROJECT_ROOT / "backend" / "data"

DATABASE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATABASE URL
# ============================================================

DATABASE_PATH = DATABASE_DIR / "media_forensics.db"

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"


# ============================================================
# DATABASE ENGINE
# ============================================================

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False
    }
)


# ============================================================
# SESSION
# ============================================================

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


# ============================================================
# BASE MODEL
# ============================================================

Base = declarative_base()


# ============================================================
# DATABASE SESSION
# ============================================================

def get_db():
    """
    Create a database session.

    The session is automatically closed
    after use.
    """

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


# ============================================================
# CREATE DATABASE
# ============================================================

def create_database():
    """
    Create all VERITAS database tables.

    This is safe to call every time the application starts.
    Existing tables are not deleted.
    """

    # Import models here so SQLAlchemy knows about them
    # before create_all() is executed.
    from backend.models.analysis import (
        Analysis,
        PasskeyCredential,
    )

    Base.metadata.create_all(
        bind=engine
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    create_database()

    print()
    print("=" * 60)
    print("DATABASE CREATED SUCCESSFULLY")
    print("=" * 60)

    print()
    print("Database location:")
    print(DATABASE_PATH)