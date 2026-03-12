# ============================================================
# LPR FastAPI Application Entry Point (with Licensing)
# ============================================================

from fastapi import FastAPI
import threading
import logging
import sys
from datetime import datetime, timezone

from app.camera_worker import CameraWorker
from app.logging_config import setup_logging
from app.config import *
from verify_license import verify_license, LicenseError


from contextlib import asynccontextmanager


# ============================================================
# Logging Initialization
# ============================================================
setup_logging(LOG_LEVEL)
logging.info("Starting LPR FastAPI Service...")


# ============================================================
# License Validation & Watchdog
# ============================================================

# Will store license expiry datetime
LICENSE_EXPIRES_AT = None

def license_watchdog():
    """
    Continuously checks license expiry.
    If expired → force shutdown immediately.
    Runs independently in background daemon thread.
    """
    global LICENSE_EXPIRES_AT
    while True:
        if LICENSE_EXPIRES_AT:
            now = datetime.now(timezone.utc)
            if now > LICENSE_EXPIRES_AT:
                logging.error("❌ License expired during runtime. Shutting down.")
                sys.exit(1)
        threading.Event().wait(5)  # Check every 5 seconds

# Start watchdog daemon thread immediately
watchdog_thread = threading.Thread(target=license_watchdog, daemon=True, name="LicenseWatchdog")
watchdog_thread.start()



worker = CameraWorker()
LICENSE_EXPIRES_AT = None  # Global for watchdog

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler:

    1. Perform license validation (blocking)
    2. Set LICENSE_EXPIRES_AT for watchdog
    3. Start CameraWorker in background daemon thread
    4. Optionally handle shutdown
    """
    global LICENSE_EXPIRES_AT

    # -------- License Validation --------
    logging.info("✅ Performing license validation...")

    try:
        license_data = verify_license()
        expires_at_str = license_data.get("expires_at")
        LICENSE_EXPIRES_AT = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))

        logging.info(f"License valid")
        logging.info(f"Licensed user: {license_data.get('user_name')}")
    except LicenseError as e:
        logging.error(f"❌ LICENSE ERROR: {e}")
        import sys
        sys.exit(1)

    # -------- Camera Worker --------
    logging.info("Starting CameraWorker background thread...")
    thread = threading.Thread(
        target=worker.run,
        daemon=True,
        name="CameraWorkerThread"
    )
    thread.start()
    logging.info("CameraWorker thread started successfully.")

    # Yield control to FastAPI (app runs here)
    yield

    # -------- Optional Shutdown --------
    logging.info("Shutting down LPR service...")
    # Add any worker cleanup logic here if needed



# ============================================================
# FastAPI App Initialization
# ============================================================
app = FastAPI(
    title="LPR Service",
    description="License Plate Recognition GPU Service",
    version="1.0.0", lifespan=lifespan
)


# ============================================================
# Health Check Endpoint
# ============================================================
@app.get("/health")
def health():
    logging.debug("Health check requested.")
    return {
        "status": "running",
        "service": "lpr",
        "version": "1.0.0"
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)