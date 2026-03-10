from ultralytics import YOLO
import numpy as np
import logging
from app.config import OCR_SUB_BATCH_SIZE

OCR_CLASSES = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"


# =====================================================================
# LINE GROUPING — Y-interval overlap (tilt-tolerant, padding-immune)
# =====================================================================

def group_boxes_into_lines(boxes, overlap_threshold: float = 0.4):
    """
    Groups character bounding boxes into text lines using Y-interval overlap.

    Two character boxes are considered to be on the same line if their
    vertical extents overlap by at least `overlap_threshold` fraction of
    the shorter box's height.  This is robust to:
        - Padding in the plate crop  (no chars detected there → irrelevant)
        - Moderate plate tilt        (same-line chars still share a Y band)
        - Low-confidence phantom boxes (filtered out up front)

    Returns a list of groups (each group = list of boxes), ordered
    top-to-bottom; within each group boxes are ordered left-to-right.
    At most 2 groups are returned — if more are found (glare / dirt),
    only the two largest by character count are kept.
    """
    # 1. Drop very low-confidence detections (noise / glare)
    boxes = [b for b in boxes if float(b.conf[0]) >= 0.35]
    if not boxes:
        return []

    # 2. Pre-sort by vertical centre to make grouping deterministic
    boxes = sorted(
        boxes,
        key=lambda b: (float(b.xyxy[0][1]) + float(b.xyxy[0][3])) / 2.0
    )

    groups = []   # list of lists of boxes

    for box in boxes:
        b_y1 = float(box.xyxy[0][1])
        b_y2 = float(box.xyxy[0][3])
        b_h  = max(b_y2 - b_y1, 1.0)

        placed = False
        for group in groups:
            # Current Y span of the group
            g_y1 = min(float(b.xyxy[0][1]) for b in group)
            g_y2 = max(float(b.xyxy[0][3]) for b in group)

            overlap  = min(b_y2, g_y2) - max(b_y1, g_y1)
            min_span = min(b_h, max(g_y2 - g_y1, 1.0))
            ratio    = overlap / min_span

            if ratio >= overlap_threshold:
                group.append(box)
                placed = True
                break

        if not placed:
            groups.append([box])   # start a new line group

    # 3. Guard: keep only the 2 largest groups (discard phantom lines)
    if len(groups) > 2:
        groups = sorted(groups, key=len, reverse=True)[:2]

    # 4. Sort groups top-to-bottom by mean Y
    groups.sort(key=lambda g: sum(
        (float(b.xyxy[0][1]) + float(b.xyxy[0][3])) / 2.0 for b in g
    ) / len(g))

    # 5. Within each group sort left-to-right by X centre
    for g in groups:
        g.sort(key=lambda b: (float(b.xyxy[0][0]) + float(b.xyxy[0][2])) / 2.0)

    return groups


class OCR:
    """
    Production-grade OCR class for license plates.

    Features:
        - Two-line / single-line plate detection via Y-interval overlap grouping
        - Tilt-tolerant character ordering
        - Class index safety validation
        - Minimum character filtering
        - Confidence averaging
        - Sub-batching to prevent CUDA OOM
    """

    def __init__(self, model_path: str, device: str):
        self.model = YOLO(model_path)
        self.model.to(device)
        logging.info(f"[OCR] Loaded model {model_path} on {device}")

    def read_with_confidence(self, plate_img) -> tuple[str, float]:
        """
        Reads a single plate image and returns (plate_text, avg_confidence).
        Handles both single-line and two-line plates automatically.
        """
        try:
            results = self.model.predict(plate_img, conf=0.3, verbose=False)
        except Exception as e:
            logging.exception(f"[OCR ERROR] Model inference failed: {e}")
            return "", 0.0

        if not results or len(results[0].boxes) == 0:
            return "", 0.0

        # Group boxes into lines (handles tilt + padding + two-line plates)
        groups = group_boxes_into_lines(results[0].boxes)

        chars = []
        confs = []

        for group in groups:               # top line first, then bottom line
            for box in group:
                cid  = int(box.cls[0])
                conf = float(box.conf[0])
                if cid < 0 or cid >= len(OCR_CLASSES):
                    logging.warning(f"[OCR] Invalid class id: {cid}")
                    continue
                chars.append(OCR_CLASSES[cid])
                confs.append(conf)

        if len(chars) < 4:
            logging.debug("[OCR] Too few characters detected. Skipping.")
            return "", 0.0

        text     = "".join(chars)
        avg_conf = float(np.mean(confs)) if confs else 0.0

        n_lines = len(groups)
        logging.debug(
            f"[OCR] Plate={text}  Lines={n_lines}  "
            f"Chars={len(chars)}  AvgConf={avg_conf:.2f}"
        )

        return text, avg_conf

    def read_batch_with_confidence(self, plate_images: list) -> list:
        """
        Batch OCR: accepts a list of plate image arrays.
        Runs YOLO inference in chunks of OCR_SUB_BATCH_SIZE.
        Handles both single-line and two-line plates automatically.

        Returns:
            list of (text, confidence) tuples, one per input image.
        """
        if not plate_images:
            return []

        all_results = []

        for i in range(0, len(plate_images), OCR_SUB_BATCH_SIZE):
            chunk = plate_images[i:i + OCR_SUB_BATCH_SIZE]
            try:
                chunk_results = self.model.predict(chunk, conf=0.3, verbose=False)
                all_results.extend(chunk_results)
            except Exception as e:
                logging.exception(
                    f"[OCR BATCH ERROR] Inference failed for chunk at {i}: {e}"
                )
                all_results.extend([None] * len(chunk))

        output = []

        for res in all_results:
            if not res or len(res.boxes) == 0:
                output.append(("", 0.0))
                continue

            # Group boxes into lines (handles tilt + padding + two-line plates)
            groups = group_boxes_into_lines(res.boxes)

            chars = []
            confs = []

            for group in groups:           # top line first, then bottom line
                for box in group:
                    cid  = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cid < 0 or cid >= len(OCR_CLASSES):
                        logging.warning(f"[OCR BATCH] Invalid class id: {cid}")
                        continue
                    chars.append(OCR_CLASSES[cid])
                    confs.append(conf)

            if len(chars) < 4:
                output.append(("", 0.0))
                continue

            text     = "".join(chars)
            avg_conf = float(np.mean(confs)) if confs else 0.0

            n_lines = len(groups)
            logging.debug(
                f"[OCR BATCH] Plate={text}  Lines={n_lines}  "
                f"Chars={len(chars)}  AvgConf={avg_conf:.2f}"
            )

            output.append((text, avg_conf))

        return output
