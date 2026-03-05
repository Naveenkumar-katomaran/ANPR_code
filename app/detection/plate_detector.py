import cv2
import numpy as np
import logging

class PlateDetector:
    def __init__(self, cfg, weights, device="cpu"):
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        if device == "cuda":
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                # Test the backend with a dummy forward pass to catch the assertion early
                blob = np.zeros((1, 3, 416, 416), dtype=np.float32)
                self.net.setInput(blob)
                self.net.forward()
                logging.info("[PLATE DETECT] Successfully initialized OpenCV CUDA backend")
            except Exception as e:
                logging.warning(f"[PLATE DETECT] OpenCV CUDA not supported or failed: {e}. Falling back to CPU.")
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        h, w = frame.shape[:2]
        boxes = []

        for out in outputs:
            for det in out:
                scores = det[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.6:
                    cx, cy, bw, bh = (
                        int(det[0]*w),
                        int(det[1]*h),
                        int(det[2]*w),
                        int(det[3]*h),
                    )
                    x1 = int(cx - bw/2)
                    y1 = int(cy - bh/2)
                    x2 = x1 + bw
                    y2 = y1 + bh
                    boxes.append((x1,y1,x2,y2,confidence))

        return boxes
