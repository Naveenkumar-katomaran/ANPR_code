import json
import base64
from pathlib import Path
from datetime import datetime, timezone
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class LicenseError(Exception):
    pass


def verify_license():
    import sys

    # -----------------------------
    # Detect runtime environment
    # -----------------------------
    if getattr(sys, "frozen", False):
        # Running as PyInstaller binary
        base_dir = Path(sys.executable).parent
    else:
        base_dir = Path(__file__).parent

    # -----------------------------
    # Detect if running inside Docker
    # -----------------------------
    docker_license = Path("../license/license.json")
    local_license = base_dir / "license" / "license.json"

    if docker_license.exists():
        license_file = docker_license
    else:
        license_file = local_license

    # Public key always next to app (not in license folder)
    public_key_file = base_dir / "public_key.pem"

    # -----------------------------
    # Validate files
    # -----------------------------
    if not license_file.exists():
        raise LicenseError(f"License file missing: {license_file}")

    if not public_key_file.exists():
        raise LicenseError(f"Public key missing: {public_key_file}")

    # -----------------------------
    # Load license
    # -----------------------------
    with open(license_file, "r") as f:
        data = json.load(f)

    payload = data.get("payload")
    signature_b64 = data.get("signature")

    if not payload or not signature_b64:
        raise LicenseError("License payload or signature missing")

    signature = base64.b64decode(signature_b64)

    payload_bytes = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True
    ).encode()

    # -----------------------------
    # Load public key
    # -----------------------------
    with open(public_key_file, "rb") as f:
        public_key = serialization.load_pem_public_key(f.read())

    # -----------------------------
    # Verify signature
    # -----------------------------
    try:
        public_key.verify(
            signature,
            payload_bytes,
            padding.PKCS1v15(),
            hashes.SHA256()
        )
    except Exception:
        raise LicenseError("Invalid license signature")

    # -----------------------------
    # Expiration check
    # -----------------------------
    expires_at_str = payload.get("expires_at")
    if not expires_at_str:
        raise LicenseError("License expiration missing")

    expires_at = datetime.fromisoformat(
        expires_at_str.replace("Z", "+00:00")
    )

    if datetime.now(timezone.utc) > expires_at:
        raise LicenseError("License expired")

    return payload
