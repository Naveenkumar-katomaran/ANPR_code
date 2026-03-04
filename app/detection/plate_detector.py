import cv2
import numpy as np

class PlateDetector:
    def __init__(self, cfg, weights):
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)

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
