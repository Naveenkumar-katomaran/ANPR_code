import cv2
import numpy as np
import logging
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, cfg, weights, device="cpu"):
        self.device = device
        self.use_yolo = weights.endswith(".pt")
        
        if self.use_yolo:
            logging.info(f"[PLATE DETECT] Loading YOLOv8 model from {weights}")
            self.model = YOLO(weights)
            self.model.to(self.device)
        else:
            logging.info(f"[PLATE DETECT] Loading Darknet model: {cfg}, {weights}")
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

    def detect(self, frame, conf_thresh: float = 0.25, nms_thresh: float = 0.4):
        if self.use_yolo:
            results = self.model(frame, conf=conf_thresh, iou=nms_thresh, imgsz=640, verbose=False)
            detections = []
            if len(results) > 0 and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    detections.append((x1, y1, x2, y2, conf))
            return detections
        else:
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True)
            self.net.setInput(blob)
            outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

            h, w = frame.shape[:2]
            raw_boxes    = []   # [x, y, w, h] for NMS
            raw_confs    = []   # confidence scores for NMS
            raw_xyxy     = []   # (x1, y1, x2, y2) for final output

            for out in outputs:
                for det in out:
                    objectness = float(det[4])          # YOLO objectness score
                    scores     = det[5:]
                    class_id   = int(np.argmax(scores))
                    class_conf = float(scores[class_id])
                    confidence = objectness * class_conf  # standard YOLO confidence

                    if confidence >= conf_thresh:
                        cx = int(det[0] * w)
                        cy = int(det[1] * h)
                        bw = int(det[2] * w)
                        bh = int(det[3] * h)
                        x1 = max(int(cx - bw / 2), 0)
                        y1 = max(int(cy - bh / 2), 0)
                        raw_boxes.append([x1, y1, bw, bh])
                        raw_confs.append(confidence)
                        raw_xyxy.append((x1, y1, x1 + bw, y1 + bh))

            if not raw_boxes:
                return []

            # Apply Non-Maximum Suppression
            indices = cv2.dnn.NMSBoxes(raw_boxes, raw_confs, conf_thresh, nms_thresh)
            if len(indices) == 0:
                return []

            results_list = []
            for i in indices.flatten():
                x1, y1, x2, y2 = raw_xyxy[i]
                results_list.append((x1, y1, x2, y2, raw_confs[i]))

            logging.debug(f"[PLATE DETECT] Found {len(results_list)} plate(s) after NMS")
            return results_list
