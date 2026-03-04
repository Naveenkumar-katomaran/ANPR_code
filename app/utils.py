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
    logging.info(f"[RTSP] Opening stream: {RTSP_URL}")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    for attempt in range(3):
        if cap.isOpened():
            logging.info("[RTSP] Stream opened successfully")
            break
        logging.warning(f"[RTSP] Attempt {attempt+1}/3 failed, retrying...")
        cap.release()
        time.sleep(2)
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        logging.error("[RTSP] Failed to open stream")
        return None

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap