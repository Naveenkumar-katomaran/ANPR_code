import os
import sys
import json
from dotenv import load_dotenv

# -----------------------------
# Resolve BASE_DIR for PyInstaller
# -----------------------------
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Load .env
# -----------------------------
ENV_PATH = os.path.join(os.path.dirname(BASE_DIR), ".env")
load_dotenv(dotenv_path=ENV_PATH)
print(f"Loading .env from: {ENV_PATH}")




SHOW_LPR_WINDOW = os.getenv("SHOW_LPR_WINDOW", "False").lower() in ["1", "true", "yes"]

RTSP_URL = os.getenv("RTSP_URL")
CAMERA_ID = os.getenv("CAMERA_ID")

API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")

LOG_FILE = os.getenv("LOG_FILE","application.log")

DUPLICATE_WINDOW_SECONDS = int(os.getenv("DUPLICATE_WINDOW_SECONDS", 30))

VEHICLE_SESSION_TIMEOUT = int(os.getenv("VEHICLE_SESSION_TIMEOUT", 5))
MAX_SESSION_DURATION_SECONDS = int(os.getenv("MAX_SESSION_DURATION_SECONDS", 30))
MIN_FRAMES_BEFORE_DECISION = int(os.getenv("MIN_FRAMES_BEFORE_DECISION", 3))
MIN_OCR_CONFIDENCE = float(os.getenv("MIN_OCR_CONFIDENCE", 0.6))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.3))


FRAME_SKIP = int(os.getenv("FRAME_SKIP", 3))              # Decode 1 frame every N
RESIZE_WIDTH = int(os.getenv("RESIZE_WIDTH", 1280))       # Resize for
MAX_VEHICLES_PER_FRAME = int(os.getenv("MAX_VEHICLES_PER_FRAME", 3))  # Avoid overload
MAX_BATCH_IMAGES = int(os.getenv("MAX_BATCH_IMAGES", 60))  # Max plate crops saved per session
OCR_SUB_BATCH_SIZE = int(os.getenv("OCR_SUB_BATCH_SIZE", 30))  # Chunk size for sub-batch OCR

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

DEVICE = os.getenv("DEVICE", "cpu").lower()  # "cpu" or "cuda"
STORE_SESSION_PROOF = os.getenv("STORE_SESSION_PROOF", "True").lower() in ["1", "true", "yes"]
COLLECT_SESSION_IMAGE = os.getenv("COLLECT_SESSION_IMAGE", "False").lower() in ["1", "true", "yes"]


TOTAL_CANDIDATES_COUNT = int(os.getenv("TOTAL_CANDIDATES_COUNT", 20))

VALIDATE_INDIAN_PLATE = os.getenv("VALIDATE_INDIAN_PLATE", "True").lower() in ["1", "true", "yes"]

PORT = int(os.getenv("PORT", 7000))

# -----------------------------
# Direction filter
# -----------------------------
# ENTRY  : only process vehicles crossing from right-of-line → left-of-line (P1→P2)
# EXIT   : only process vehicles crossing the opposite way
# BOTH   : process all vehicles regardless of crossing direction
DIRECTION = os.getenv("DIRECTION", "BOTH").upper()
if DIRECTION not in ("ENTRY", "EXIT", "BOTH"):
    print(f"[CONFIG] WARNING: DIRECTION='{DIRECTION}' is invalid. Defaulting to BOTH.")
    DIRECTION = "BOTH"


# -----------------------------
# Camera line loader
# -----------------------------
_CAMERAS_JSON = os.path.join(os.path.dirname(BASE_DIR), "config", "cameras.json")

def load_camera_line(source_url: str):
    """
    Load the virtual crossing line for *source_url* from config/cameras.json.

    Returns (x1, y1, x2, y2, native_w, native_h) where the coordinates are
    in the native frame resolution that was used when the line was drawn.
    The caller is responsible for scaling to the actual capture resolution.

    Raises SystemExit(1) if the file is missing or the source has no entry,
    so the worker refuses to start with a clear log message.
    """
    if not os.path.exists(_CAMERAS_JSON):
        print(
            f"[CONFIG] ERROR: cameras.json not found at {_CAMERAS_JSON}. "
            f"Run  python tools/setup_line.py  first."
        )
        sys.exit(1)

    with open(_CAMERAS_JSON) as f:
        data = json.load(f)

    if source_url not in data:
        print(
            f"[CONFIG] ERROR: No line configured for source '{source_url}'. "
            f"Run  python tools/setup_line.py  to set it up."
        )
        sys.exit(1)

    entry = data[source_url]
    ln = entry["line"]
    native_w = int(entry.get("frame_width",  ln["x2"]))   # fallback: use x2 as width estimate
    native_h = int(entry.get("frame_height", ln["y1"]))   # fallback: use y1 as height estimate
    coords = (int(ln["x1"]), int(ln["y1"]), int(ln["x2"]), int(ln["y2"]), native_w, native_h)
    print(
        f"[CONFIG] Line loaded for '{source_url}': "
        f"P1=({coords[0]},{coords[1]})  P2=({coords[2]},{coords[3]})  "
        f"native_res={native_w}×{native_h}"
    )
    return coords
