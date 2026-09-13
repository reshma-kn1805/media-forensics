import json


from fastapi import APIRouter

from webauthn import (
    generate_authentication_options,
    generate_registration_options,
    options_to_json,
    verify_authentication_response,
    verify_registration_response,
)

from webauthn.helpers.structs import (
    PublicKeyCredentialDescriptor,
)

from backend.database import SessionLocal
from backend.models.analysis import PasskeyCredential


router = APIRouter(
    prefix="/auth",
    tags=["WebAuthn Authentication"]
)


# ============================================================
# WEBAUTHN CONFIGURATION
# ============================================================

RP_NAME = "VERITAS"
RP_ID = "localhost"
ORIGIN = "http://localhost:5173"

DEMO_USER_ID = "veritas-analyst"
DEMO_USERNAME = "analyst"


# ============================================================
# TEMPORARY CHALLENGE STORAGE
# ============================================================

registration_challenges = {}

authentication_challenges = {}


# ============================================================
# STATUS
# ============================================================

@router.get("/status")
def auth_status():
    return {
        "status": "ready",
        "authentication": "WebAuthn / Passkey",
        "rp_name": RP_NAME,
        "rp_id": RP_ID,
        "origin": ORIGIN,
        "user": DEMO_USERNAME,
    }


# ============================================================
# PASSKEY REGISTRATION OPTIONS
# ============================================================

@router.post("/register/options")
def registration_options():
    """
    Generate WebAuthn registration options.
    """

    user_id = DEMO_USER_ID.encode("utf-8")

    options = generate_registration_options(
        rp_id=RP_ID,
        rp_name=RP_NAME,
        user_id=user_id,
        user_name=DEMO_USERNAME,
        user_display_name="VERITAS Analyst",
    )

    registration_challenges[
        DEMO_USERNAME
    ] = options.challenge

    return json.loads(
        options_to_json(options)
    )


# ============================================================
# PASSKEY REGISTRATION VERIFICATION
# ============================================================

@router.post("/register/verify")
def registration_verify(credential: dict):
    """
    Verify a newly created WebAuthn passkey and store its
    public credential information in the database.
    """

    expected_challenge = registration_challenges.get(
        DEMO_USERNAME
    )

    if expected_challenge is None:
        return {
            "success": False,
            "message": "No active registration challenge found."
        }

    db = SessionLocal()

    try:

        # ----------------------------------------------------
        # VERIFY REGISTRATION
        # ----------------------------------------------------

        verification = verify_registration_response(
            credential=credential,
            expected_challenge=expected_challenge,
            expected_rp_id=RP_ID,
            expected_origin=ORIGIN,
        )

        credential_id_hex = (
            verification.credential_id.hex()
        )

        # ----------------------------------------------------
        # CHECK DUPLICATE
        # ----------------------------------------------------

        existing_credential = (
            db.query(PasskeyCredential)
            .filter(
                PasskeyCredential.credential_id
                == credential_id_hex
            )
            .first()
        )

        if existing_credential is not None:
            return {
                "success": False,
                "message": "This passkey is already registered."
            }

        # ----------------------------------------------------
        # STORE CREDENTIAL
        # ----------------------------------------------------

        passkey = PasskeyCredential(
            username=DEMO_USERNAME,

            credential_id=credential_id_hex,

            credential_public_key=(
                verification.credential_public_key.hex()
            ),

            sign_count=verification.sign_count,

            aaguid=str(
                verification.aaguid
            ),

            credential_type=str(
                verification.credential_type
            ),

            device_type=str(
                verification.credential_device_type
            ),

            credential_backed_up=(
                1
                if verification.credential_backed_up
                else 0
            ),

            user_verified=(
                1
                if verification.user_verified
                else 0
            ),
        )

        db.add(passkey)
        db.commit()
        db.refresh(passkey)

        registration_challenges.pop(
            DEMO_USERNAME,
            None
        )

        return {
            "success": True,
            "message": (
                "Passkey registered, verified, "
                "and securely stored."
            ),
            "credential_id": credential_id_hex,
        }

    except Exception as error:

        db.rollback()

        return {
            "success": False,
            "message": (
                f"Passkey verification failed: {error}"
            ),
        }

    finally:

        db.close()


# ============================================================
# PASSKEY LOGIN OPTIONS
# ============================================================

@router.post("/login/options")
def login_options():
    """
    Generate WebAuthn authentication options for the
    registered VERITAS analyst passkey.
    """

    db = SessionLocal()

    try:

        # ----------------------------------------------------
        # FIND REGISTERED PASSKEYS
        # ----------------------------------------------------

        passkeys = (
            db.query(PasskeyCredential)
            .filter(
                PasskeyCredential.username
                == DEMO_USERNAME
            )
            .all()
        )

        if not passkeys:
            return {
                "success": False,
                "message": "No registered passkey found."
            }

        # ----------------------------------------------------
        # ALLOWED CREDENTIALS
        # ----------------------------------------------------

        allow_credentials = [
            PublicKeyCredentialDescriptor(
                id=bytes.fromhex(
                    passkey.credential_id
                )
            )
            for passkey in passkeys
        ]

        # ----------------------------------------------------
        # GENERATE AUTHENTICATION OPTIONS
        # ----------------------------------------------------

        options = generate_authentication_options(
            rp_id=RP_ID,
            allow_credentials=allow_credentials,
        )

        authentication_challenges[
            DEMO_USERNAME
        ] = options.challenge

        return json.loads(
            options_to_json(options)
        )

    finally:

        db.close()


# ============================================================
# PASSKEY LOGIN VERIFICATION
# ============================================================

@router.post("/login/verify")
def login_verify(credential: dict):
    """
    Verify a WebAuthn authentication response.

    The server looks up the registered credential's public
    key and current sign count, then verifies the signature
    returned by the authenticator.
    """

    expected_challenge = authentication_challenges.get(
        DEMO_USERNAME
    )

    if expected_challenge is None:
        return {
            "success": False,
            "message": "No active authentication challenge found."
        }

    db = SessionLocal()

    try:

        # ----------------------------------------------------
        # GET CREDENTIAL ID FROM BROWSER RESPONSE
        # ----------------------------------------------------

        raw_id = credential.get("rawId")

        if not raw_id:
            return {
                "success": False,
                "message": "Authentication credential ID is missing."
            }

        # ----------------------------------------------------
        # FIND REGISTERED CREDENTIAL
        # ----------------------------------------------------

        credential_id_hex = bytes.fromhex(
            __import__("base64").urlsafe_b64decode(
                raw_id + "=" * (
                    4 - len(raw_id) % 4
                ) % 4
            ).hex()
        ).hex()

        passkey = (
            db.query(PasskeyCredential)
            .filter(
                PasskeyCredential.credential_id
                == credential_id_hex
            )
            .first()
        )

        if passkey is None:
            return {
                "success": False,
                "message": "Registered passkey was not found."
            }

        # ----------------------------------------------------
        # VERIFY AUTHENTICATION RESPONSE
        # ----------------------------------------------------

        verification = verify_authentication_response(
            credential=credential,
            expected_challenge=expected_challenge,
            expected_rp_id=RP_ID,
            expected_origin=ORIGIN,
            credential_public_key=bytes.fromhex(
                passkey.credential_public_key
            ),
            credential_current_sign_count=(
                passkey.sign_count
            ),
        )

        # ----------------------------------------------------
        # UPDATE SIGN COUNT
        # ----------------------------------------------------

        passkey.sign_count = (
            verification.new_sign_count
        )

        passkey.user_verified = (
            1
            if verification.user_verified
            else 0
        )

        db.commit()

        # ----------------------------------------------------
        # REMOVE USED CHALLENGE
        # ----------------------------------------------------

        authentication_challenges.pop(
            DEMO_USERNAME,
            None
        )

        # ----------------------------------------------------
        # AUTHENTICATION SUCCESS
        # ----------------------------------------------------

        return {
            "success": True,
            "message": "Passkey authentication successful.",
            "username": DEMO_USERNAME,
            "user_verified": verification.user_verified,
            "sign_count": verification.new_sign_count,
        }

    except Exception as error:

        db.rollback()

        return {
            "success": False,
            "message": (
                f"Passkey authentication failed: {error}"
            ),
        }

    finally:

        db.close()