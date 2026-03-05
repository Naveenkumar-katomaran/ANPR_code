import cv2
import numpy as np
import os
import logging
import time

def sharpness_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def safe_crop(img, box):
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    return img[y1:y2, x1:x2]



def open_capture(RTSP_URL: str):
    """
    Open the RTSP or video stream with retries and low-latency tuning.
    """

    logging.info(f"[RTSP] Attempting to open stream: {RTSP_URL}")

    # ---------- FFmpeg low-latency options (ffplay equivalent) ----------
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|"
        # "fflags;nobuffer|"
        # "flags;low_delay|"
        # "analyzeduration;0|"
        # "probesize;32"
    )

    # ---------- Local webcam fallback ----------
    if RTSP_URL in ["0", None, ""]:
        logging.info("[RTSP] Opening local webcam (0)")
        cap = cv2.VideoCapture(0)
        return cap if cap.isOpened() else None

    # ---------- HTTP / recordings ----------
    if RTSP_URL.startswith(("http://", "https://")) or ("recordings" in RTSP_URL):
        logging.info("[RTSP] Opening HTTP/HTTPS stream")
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        return cap if cap.isOpened() else None

    url = RTSP_URL
    cap = None

    # ---------- Try FFmpeg backend (3 attempts) ----------
    for attempt in range(1, 4):
        logging.info(f"[RTSP] Open attempt {attempt}/3 using FFmpeg backend...")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        if cap.isOpened():
            break

        logging.warning("[RTSP] FFmpeg backend failed, retrying...")
        cap.release()
        time.sleep(2)

    # ---------- Fallback ----------
    if not cap or not cap.isOpened():
        logging.warning("[RTSP] Falling back to CAP_ANY backend...")
        cap = cv2.VideoCapture(url, cv2.CAP_ANY)

    if not cap.isOpened():
        logging.error("[RTSP] FAILED to open RTSP stream with OpenCV.")
        cap.release()
        return None

    # ---------- Low-latency tuning ----------
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Optional timeout support (depends on OpenCV build)
    try:
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
    except Exception as e:
        logging.debug(f"[RTSP] Timeout props not supported: {e}")

    # ---------- Stream info ----------
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_str = f"{fps:.1f}" if fps and fps > 0 else "unknown"

    logging.info(f"[RTSP] SUCCESS! Stream opened: {w}x{h} @ {fps_str} FPS")

    return cap