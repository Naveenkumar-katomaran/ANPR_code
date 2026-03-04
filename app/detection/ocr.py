# from ultralytics import YOLO
# import numpy as np
# import logging

# OCR_CLASSES = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

# class OCR:
#     """
#     OCR class for reading license plates using YOLOv8.
#     Provides both simple read() and read_with_confidence().
#     """

#     def __init__(self, model_path: str, device: str):
#         self.model = YOLO(model_path)
#         self.model.to(device)
#         logging.info(f"[OCR] Loaded model {model_path} on {device}")

#     def read(self, plate_img) -> str:
#         """
#         Simple OCR: returns text only.

#         Args:
#             plate_img (np.array): Plate crop image.

#         Returns:
#             str: Recognized plate text
#         """
#         results = self.model.predict(plate_img, conf=0.7, verbose=False)
#         chars = []

#         if results and len(results[0].boxes) > 0:
#             boxes = sorted(results[0].boxes, key=lambda b: b.xyxy[0][0])
#             for box in boxes:
#                 cid = int(box.cls[0])
#                 chars.append(OCR_CLASSES[cid])
#         return "".join(chars)

#     def read_with_confidence(self, plate_img) -> tuple[str, float]:
#         """
#         Advanced OCR: returns text and confidence score.

#         Args:
#             plate_img (np.array): Plate crop image.

#         Returns:
#             tuple: (text: str, confidence: float)
#                 - text: recognized string
#                 - confidence: average confidence of all characters (0-1)
#         """
#         results = self.model.predict(plate_img, conf=0.3, verbose=False)  # lower conf to capture all chars
#         if not results or len(results[0].boxes) == 0:
#             return "", 0.0

#         boxes = results[0].boxes
#         # Sort left to right
#         boxes = sorted(boxes, key=lambda b: b.xyxy[0][0])

#         chars = []
#         confs = []

#         for box in boxes:
#             cid = int(box.cls[0])
#             conf = float(box.conf[0])  # confidence for this box
#             chars.append(OCR_CLASSES[cid])
#             confs.append(conf)

#         text = "".join(chars)
#         avg_conf = np.mean(confs) if confs else 0.0

#         logging.debug(f"[OCR] Recognized plate: {text} AvgConf={avg_conf:.2f}")
#         return text, avg_conf










from ultralytics import YOLO
import numpy as np
import logging
from app.config import OCR_SUB_BATCH_SIZE

OCR_CLASSES = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"


class OCR:
    """
    Production-grade OCR class for license plates.

    Features:
        - Left-to-right character sorting
        - Class index safety validation
        - Minimum character filtering
        - Confidence averaging
        - Clean fail-safe behavior
        - Sub-batching to prevent CUDA OOM
    """

    def __init__(self, model_path: str, device: str):
        self.model = YOLO(model_path)
        self.model.to(device)
        logging.info(f"[OCR] Loaded model {model_path} on {device}")

    def read_with_confidence(self, plate_img) -> tuple[str, float]:
        """
        Reads plate image and returns:
            (plate_text, avg_confidence)

        Returns empty string if:
            - No detections
            - Too few characters
        """

        try:
            results = self.model.predict(
                plate_img,
                conf=0.3,  # lower threshold for recall
                verbose=False
            )
        except Exception as e:
            logging.exception(f"[OCR ERROR] Model inference failed: {e}")
            return "", 0.0

        if not results or len(results[0].boxes) == 0:
            return "", 0.0

        boxes = results[0].boxes

        # Sort left to right
        boxes = sorted(boxes, key=lambda b: float(b.xyxy[0][0]))

        chars = []
        confs = []

        for box in boxes:
            cid = int(box.cls[0])
            conf = float(box.conf[0])

            # Safety check
            if cid < 0 or cid >= len(OCR_CLASSES):
                logging.warning(f"[OCR] Invalid class id: {cid}")
                continue

            chars.append(OCR_CLASSES[cid])
            confs.append(conf)

        # Minimum character filter (avoid noise)
        if len(chars) < 4:
            logging.debug("[OCR] Too few characters detected. Skipping.")
            return "", 0.0

        text = "".join(chars)
        avg_conf = float(np.mean(confs)) if confs else 0.0

        logging.debug(
            f"[OCR] Plate={text} "
            f"Chars={len(chars)} "
            f"AvgConf={avg_conf:.2f}"
        )

        return text, avg_conf

    def read_batch_with_confidence(self, plate_images: list) -> list:
        """
        Batch OCR: accepts a list of plate image arrays.
        Runs multiple YOLO inference passes in chunks of OCR_SUB_BATCH_SIZE.

        Returns:
            list of (text, confidence) tuples, one per input image.
        """
        if not plate_images:
            return []

        all_results = []

        # Process in sub-batches to prevent CUDA OOM
        for i in range(0, len(plate_images), OCR_SUB_BATCH_SIZE):
            chunk = plate_images[i:i + OCR_SUB_BATCH_SIZE]
            try:
                chunk_results = self.model.predict(
                    chunk,
                    conf=0.3,
                    verbose=False
                )
                all_results.extend(chunk_results)
            except Exception as e:
                logging.exception(f"[OCR BATCH ERROR] Model inference failed for chunk starting at {i}: {e}")
                # Fill with empty results for this chunk to keep lengths aligned
                all_results.extend([None] * len(chunk))

        output = []

        for res in all_results:
            if not res or len(res.boxes) == 0:
                output.append(("", 0.0))
                continue

            boxes = sorted(res.boxes, key=lambda b: float(b.xyxy[0][0]))

            chars = []
            confs = []

            for box in boxes:
                cid = int(box.cls[0])
                conf = float(box.conf[0])

                if cid < 0 or cid >= len(OCR_CLASSES):
                    logging.warning(f"[OCR BATCH] Invalid class id: {cid}")
                    continue

                chars.append(OCR_CLASSES[cid])
                confs.append(conf)

            if len(chars) < 4:
                output.append(("", 0.0))
                continue

            text = "".join(chars)
            avg_conf = float(np.mean(confs)) if confs else 0.0

            logging.debug(
                f"[OCR BATCH] Plate={text} Chars={len(chars)} AvgConf={avg_conf:.2f}"
            )

            output.append((text, avg_conf))

        return output
