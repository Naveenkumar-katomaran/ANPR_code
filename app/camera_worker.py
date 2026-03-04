import cv2
import os
import logging
import time
import json
import uuid
import glob
import shutil
import math
from datetime import datetime
import torch

from app.config import *
from app.config import load_camera_line
from app.detection.vehicle_detector import VehicleDetector
from app.detection.plate_detector import PlateDetector
from app.detection.ocr import OCR
from app.duplicate_cache import DuplicateCache
from app.api_sender import APISender
from app.utils import safe_crop
from app.lpr.postprocess import choose_best_plate


# =====================================================
# IOU FUNCTION
# =====================================================

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = boxAArea + boxBArea - interArea
    if union == 0:
        return 0

    return interArea / union



def open_capture(RTSP_URL):
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


# =====================================================
# CAMERA WORKER
# =====================================================

class CameraWorker:

    def __init__(self):
        logging.info("[WORKER] Initializing CameraWorker")

        self.cap = open_capture(RTSP_URL)
        if not self.cap.isOpened():
            logging.error("[RTSP] Failed to open stream")

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = DEVICE
        logging.info(f"[DEVICE] Using {device}")

        self.vehicle = VehicleDetector("models/yolov8n.pt")
        self.plate = PlateDetector(
            "models/lp_detection/anprox_oloyin_tribus_minima.cfg",
            "models/lp_detection/anprox_oloyin_tribus_minima.weights"
        )
        self.ocr = OCR("models/ocr/best.pt", device)

        self.dup_cache = DuplicateCache(DUPLICATE_WINDOW_SECONDS)
        self.sender = APISender(API_URL, API_KEY, CAMERA_ID)

        # MULTI VEHICLE SESSIONS
        self.sessions = {}

        os.makedirs("outputs/plates", exist_ok=True)
        os.makedirs("outputs/sessions", exist_ok=True)
        os.makedirs("outputs/discarded", exist_ok=True)

        # Clean up any leftover temp folders from previous runs
        if os.path.exists("temp"):
            shutil.rmtree("temp")
            logging.info("[WORKER] Cleaned up leftover temp/ folder from previous run")
        os.makedirs("temp", exist_ok=True)
        os.makedirs("outputs/proofs", exist_ok=True)

        # ---- Direction line ----
        # load_camera_line() calls sys.exit(1) if not configured — intentional
        self.line = load_camera_line(RTSP_URL)   # (x1, y1, x2, y2) native coords
        logging.info(
            f"[DIRECTION] mode={DIRECTION}  "
            f"line=({self.line[0]},{self.line[1]})-({self.line[2]},{self.line[3]})"
        )

    # =====================================================
    # SESSION MANAGEMENT
    # =====================================================

    def _create_session(self, bbox):
        session_id = str(uuid.uuid4())
        now = time.time()
        temp_dir = os.path.join("temp", f"vehicle_{session_id}")

        self.sessions[session_id] = {
            "id": session_id,
            "bbox": bbox,
            "start_time": now,
            "last_seen": now,
            "temp_dir": temp_dir,
            "frame_count": 0,
            "best_plate": None,
            "best_conf": 0.0,
            "sent": False,
            # ---- Direction tracking ----
            "last_side": None,          # sign of cross-product from last frame
            "direction_crossed": None,  # "ENTRY" | "EXIT" | None
        }

        sid_short = session_id[:8]
        logging.info(
            f"[SESSION NEW]   id={sid_short}  bbox=({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})  "
            f"active_sessions={len(self.sessions)}"
        )
        return session_id

    def _match_session(self, bbox):
        best_match = None
        best_iou = 0

        for sid, session in self.sessions.items():
            iou = calculate_iou(bbox, session["bbox"])
            if iou > 0.4 and iou > best_iou:
                best_iou = iou
                best_match = sid

        if best_match:
            logging.debug(
                f"[SESSION MATCH] id={best_match[:8]}  IOU={best_iou:.2f}"
            )

        return best_match

    def _update_session(self, sid, bbox):
        """Update session bounding box and last-seen timestamp."""
        session = self.sessions.get(sid)
        if not session:
            return
        session["last_seen"] = time.time()
        session["bbox"] = bbox

    def _save_plate_image(self, sid, plate_crop, vehicle_crop=None):
        """Save a plate crop image to the session's temp folder on disk.
        Stops saving once MAX_BATCH_IMAGES is reached (CUDA OOM guard).
        """
        session = self.sessions.get(sid)
        if not session:
            return

        # ---- Cap check ----
        if session["frame_count"] >= MAX_BATCH_IMAGES:
            if session["frame_count"] == MAX_BATCH_IMAGES:
                # Log only once when cap is first hit
                logging.info(
                    f"[CAP REACHED]   session={sid[:8]}  limit={MAX_BATCH_IMAGES}  "
                    f"further crops discarded to prevent OOM"
                )
            return

        temp_dir = session["temp_dir"]
        os.makedirs(temp_dir, exist_ok=True)

        session["frame_count"] += 1
        
        # Save plate crop
        img_path = os.path.join(temp_dir, f"frame_{session['frame_count']:04d}.jpg")
        cv2.imwrite(img_path, plate_crop)
        
        # Save vehicle crop (proof) if enabled
        if STORE_SESSION_PROOF and vehicle_crop is not None:
            veh_path = os.path.join(temp_dir, f"veh_{session['frame_count']:04d}.jpg")
            cv2.imwrite(veh_path, vehicle_crop)

        logging.debug(
            f"[PLATE SAVED]   session={sid[:8]}  frame=#{session['frame_count']:04d}  "
            f"size={plate_crop.shape[1]}x{plate_crop.shape[0]}"
        )

    def _save_session_to_disk(self, session, status="finalized", discard_reason=None, plate_path=None, duplicate_blocked=False):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Route to separate folder if discarded or skipped
            if status in ["skipped", "discarded", "duplicate"]:
                folder = "outputs/discarded"
            else:
                folder = "outputs/sessions"
                
            file_path = os.path.join(folder, f"session_{timestamp}.json")

            duration = session["last_seen"] - session["start_time"] if session.get("last_seen") else 0

            data = {
                "session_id": session.get("id", "unknown"),
                "status": status,
                "discard_reason": discard_reason,
                "start_time": session["start_time"],
                "last_seen": session.get("last_seen"),
                "duration_seconds": round(duration, 2),
                "frame_count": session["frame_count"],
                "candidates": session.get("candidates", []),
                "best_plate": session.get("best_plate"),
                "best_conf": session.get("best_conf"),
                "best_plate_image": session.get("best_plate_image_path"),
                "session_proof": session.get("session_proof_path"),
                "sent": session.get("sent", False),
                "duplicate_blocked": duplicate_blocked,
                "plate_image_path": plate_path,
            }

            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)

            logging.info(f"[SESSION SAVED] {file_path}")

        except Exception as e:
            logging.error(f"[SESSION SAVE ERROR] {e}")

    def _finalize_session(self, sid):
        session = self.sessions.get(sid)
        if not session:
            return

        temp_dir = session["temp_dir"]

        def _cleanup():
            """Always remove temp folder, regardless of outcome."""
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                logging.debug(f"[TEMP CLEANED] {temp_dir}")

        # ---- Direction gate (skip before loading images) ----
        if DIRECTION != "BOTH":
            crossed = session.get("direction_crossed")
            if crossed != DIRECTION:
                reason = f"wrong_direction (crossed={crossed}, required={DIRECTION})"
                logging.info(
                    f"[DIRECTION DISCARD] session={sid[:8]}  {reason}"
                )
                self._save_session_to_disk(session, status="discarded", discard_reason=reason)
                _cleanup()
                del self.sessions[sid]
                return

        # ---- Minimum frames check ----
        image_files = sorted(glob.glob(os.path.join(temp_dir, "frame_*.jpg")))

        if len(image_files) < MIN_FRAMES_BEFORE_DECISION:
            reason = "too_few_frames"
            logging.info(
                f"[SKIP]  session={sid[:8]}  reason={reason}  "
                f"have={len(image_files)}  need>={MIN_FRAMES_BEFORE_DECISION}"
            )
            self._save_session_to_disk(session, status="skipped", discard_reason=reason)
            _cleanup()
            del self.sessions[sid]
            return

        # ---- Load images from disk ----
        images = []
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is not None:
                images.append((img_path, img))

        if not images:
            reason = "no_readable_images"
            logging.warning(f"[FINALIZE] No readable images in {temp_dir}")
            self._save_session_to_disk(session, status="discarded", discard_reason=reason)
            _cleanup()
            del self.sessions[sid]
            return

        # ---- Batch OCR ----
        ocr_t0 = time.time()
        logging.info(
            f"[BATCH OCR ▶]  session={sid[:8]}  images={len(images)}  "
            f"duration={session['last_seen'] - session['start_time']:.1f}s"
        )
        batch_results = self.ocr.read_batch_with_confidence([img for _, img in images])
        ocr_elapsed = time.time() - ocr_t0

        # ---- Build candidates from batch results ----
        all_results_count = sum(1 for t, c in batch_results if t)
        candidates = [
            {"text": text, "conf": conf}
            for text, conf in batch_results
            if text and conf >= MIN_OCR_CONFIDENCE
        ]

        logging.info(
            f"[BATCH OCR ◀]  session={sid[:8]}  time={ocr_elapsed:.2f}s  "
            f"total_reads={all_results_count}  valid_reads={len(candidates)}"
        )

        # Store candidates in session for JSON output
        session["candidates"] = candidates

        if not candidates:
            reason = "no_valid_ocr_above_threshold"
            logging.info(
                f"[DISCARD] session={sid[:8]}  reason={reason}  "
                f"min_conf={MIN_OCR_CONFIDENCE}"
            )
            self._save_session_to_disk(session, status="discarded", discard_reason=reason)
            _cleanup()
            del self.sessions[sid]
            return

        # ---- Pick best plate ----
        plate, conf = choose_best_plate(candidates)

        # Summarise what the OCR saw
        from collections import Counter
        text_freq = Counter(c["text"] for c in candidates)
        logging.info(
            f"[OCR SUMMARY]  session={sid[:8]}  readings={text_freq}  "
            f"winner='{plate}'  score={conf:.2f}"
        )

        if not plate:
            reason = "no_valid_plate_pattern"
            logging.info(f"[DISCARD] session={sid[:8]}  reason={reason}")
            self._save_session_to_disk(session, status="discarded", discard_reason=reason)
            _cleanup()
            del self.sessions[sid]
            return

        # ---- Duplicate check ----
        if self.dup_cache.is_duplicate(plate):
            logging.info(
                f"[DUPLICATE]    plate={plate}  session={sid[:8]}  "
                f"→ blocked within {DUPLICATE_WINDOW_SECONDS}s window"
            )
            self._save_session_to_disk(session, status="duplicate", discard_reason="duplicate_blocked", duplicate_blocked=True)
            _cleanup()
            del self.sessions[sid]
            return

        # ---- Save best plate image (highest confidence result) ----
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plate_path = f"outputs/plates/{plate}_{timestamp}.jpg"

        best_idx = max(
            range(len(batch_results)),
            key=lambda i: batch_results[i][1]
        )
        cv2.imwrite(plate_path, images[best_idx][1])

        # Store best image path in session for JSON output
        session["best_plate_image_path"] = plate_path

        # Handle session proof if enabled
        if STORE_SESSION_PROOF:
            best_img_path = images[best_idx][0]  # Full path: temp/vehicle_XX/frame_NNNN.jpg
            best_veh_path = best_img_path.replace("frame_", "veh_")
            
            if os.path.exists(best_veh_path):
                proof_path = f"outputs/proofs/proof_{sid[:8]}_{timestamp}.jpg"
                try:
                    shutil.copy(best_veh_path, proof_path)
                    session["session_proof_path"] = proof_path
                    logging.info(f"[PROOF SAVED]   session={sid[:8]}  path={proof_path}")
                except Exception as e:
                    logging.error(f"[PROOF ERROR]   Failed to copy {best_veh_path} to {proof_path}: {e}")
            else:
                logging.debug(f"[PROOF SKIP]    Vehicle crop not found: {best_veh_path}")
        else:
            logging.debug(f"[PROOF SKIP]    STORE_SESSION_PROOF is False")

        logging.info(
            f"[✅ FINALIZED]  plate={plate}  score={conf:.2f}  "
            f"frames={len(images)}  session={sid[:8]}  image={plate_path}"
        )

        session["best_plate"] = plate
        session["best_conf"] = conf
        self.sender.send_async(plate, plate_path)

        session["sent"] = True
        self._save_session_to_disk(session, plate_path=plate_path)

        _cleanup()
        del self.sessions[sid]

    def _cleanup_sessions(self):
        now = time.time()
        to_finalize = []

        for sid, session in self.sessions.items():
            idle = now - session["last_seen"]
            total = now - session["start_time"]

            if idle > VEHICLE_SESSION_TIMEOUT:
                logging.info(
                    f"[SESSION TIMEOUT] id={sid[:8]}  reason=idle  "
                    f"idle={idle:.1f}s  frames_collected={session['frame_count']}"
                )
                to_finalize.append(sid)
            elif total > MAX_SESSION_DURATION_SECONDS:
                logging.info(
                    f"[SESSION TIMEOUT] id={sid[:8]}  reason=max_duration  "
                    f"total={total:.1f}s  frames_collected={session['frame_count']}"
                )
                to_finalize.append(sid)

        for sid in to_finalize:
            self._finalize_session(sid)

    # =====================================================
    # DRAW INFERENCE WINDOW
    # =====================================================

    def _draw_inference_frame(self, frame, vehicle_detections):
        """
        Render a rich annotated copy of the frame for the live display.

        vehicle_detections: list of dicts:
            {
              'bbox': (x1,y1,x2,y2),
              'sid':  session_id or None,
              'plates': [ (px1,py1,px2,py2, conf, abs_x1,abs_y1,abs_x2,abs_y2) ]
            }
        """
        display = frame.copy()
        h, w = display.shape[:2]

        # ---- Colour palette ----
        COLOR_VEHICLE   = (0, 220, 0)       # bright green  – vehicle box
        COLOR_PLATE     = (0, 200, 255)     # cyan          – plate box
        COLOR_CENTROID  = (255, 80,  80)    # blue          – centroid dot
        COLOR_TEXT_BG   = (20, 20, 20)      # dark          – label background
        COLOR_TEXT      = (255, 255, 255)   # white         – label text
        COLOR_OVERLAY   = (10, 10, 10)      # very dark     – top overlay bg
        YELLOW          = (0, 220, 220)     # yellow        – line endpoints

        FONT      = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SM   = 0.45
        FONT_MD   = 0.55
        THICK_SM  = 1
        THICK_MD  = 2

        def _label(img, text, x, y, font_scale=FONT_SM, thickness=THICK_SM,
                   text_color=COLOR_TEXT, bg_color=COLOR_TEXT_BG):
            """Draw a text label with a filled background rectangle."""
            (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
            pad = 3
            cv2.rectangle(img,
                          (x - pad, y - th - pad),
                          (x + tw + pad, y + baseline + pad),
                          bg_color, cv2.FILLED)
            cv2.putText(img, text, (x, y), FONT, font_scale, text_color, thickness, cv2.LINE_AA)

        # ---- Draw each vehicle + plates ----
        for det in vehicle_detections:
            vx1, vy1, vx2, vy2 = det['bbox']
            sid   = det.get('sid')
            plates = det.get('plates', [])

            # Vehicle box
            cv2.rectangle(display, (vx1, vy1), (vx2, vy2), COLOR_VEHICLE, THICK_MD)

            # Centroid
            cx = (vx1 + vx2) // 2
            cy = (vy1 + vy2) // 2
            cv2.circle(display, (cx, cy), 5, COLOR_CENTROID, -1)

            # Session label — show direction crossing status
            if sid and sid in self.sessions:
                session  = self.sessions[sid]
                sid_short = sid[:8]
                frames   = session['frame_count']
                best_txt = session.get('best_plate') or '—'
                dir_txt  = session.get('direction_crossed') or '--'
                label    = f"ID:{sid_short}  frm:{frames}  dir:{dir_txt}  plate:{best_txt}"
            else:
                label = "NEW"

            _label(display, label, vx1, max(vy1 - 6, 14),
                   font_scale=FONT_MD, thickness=THICK_SM,
                   bg_color=(0, 120, 0))

            # Plate boxes (absolute coords on full frame)
            for plate_info in plates:
                ax1, ay1, ax2, ay2 = plate_info
                cv2.rectangle(display, (ax1, ay1), (ax2, ay2), COLOR_PLATE, THICK_MD)
                _label(display, "PLATE", ax1, max(ay1 - 4, 12),
                       font_scale=0.40, bg_color=(80, 120, 0))

        # ---- Virtual crossing line + ENTRY/EXIT arrows ----
        if self.line:
            lx1, ly1, lx2, ly2 = self.line
            cv2.line(display, (lx1, ly1), (lx2, ly2), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(display, (lx1, ly1), 6, YELLOW, -1)
            cv2.circle(display, (lx2, ly2), 6, YELLOW, -1)

            # Compute perpendicular direction arrows
            dx, dy = lx2 - lx1, ly2 - ly1
            length = math.hypot(dx, dy)
            if length > 0:
                ux, uy = dx / length, dy / length
                entry_vec = (-uy, ux)    # left of P1→P2
                exit_vec  = ( uy, -ux)  # right of P1→P2
                mid = ((lx1 + lx2) // 2, (ly1 + ly2) // 2)
                arrow_len = min(80, int(length * 0.15))

                # Entry arrow (green)
                ex = int(mid[0] + entry_vec[0] * arrow_len)
                ey = int(mid[1] + entry_vec[1] * arrow_len)
                cv2.arrowedLine(display, mid, (ex, ey), (0, 220, 60), 2,
                                tipLength=0.3, line_type=cv2.LINE_AA)
                _label(display, "ENTRY", ex + 4, ey + 4,
                       font_scale=0.45, bg_color=(0, 100, 30))

                # Exit arrow (red)
                xx_ = int(mid[0] + exit_vec[0] * arrow_len)
                xy_ = int(mid[1] + exit_vec[1] * arrow_len)
                cv2.arrowedLine(display, mid, (xx_, xy_), (60, 60, 220), 2,
                                tipLength=0.3, line_type=cv2.LINE_AA)
                _label(display, "EXIT", xx_ + 4, xy_ + 4,
                       font_scale=0.45, bg_color=(30, 0, 120))

        # ---- Top overlay: semi-transparent stats bar ----
        bar_h = 36
        # Correct alpha-blend: blend existing pixels with a dark bar
        bar_bg = display[:bar_h, :].copy()     # safe copy of the region
        cv2.rectangle(bar_bg, (0, 0), (w, bar_h), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(bar_bg, 0.75, display[:bar_h, :], 0.25, 0, display[:bar_h, :])

        active = len(self.sessions)
        veh_n  = len(vehicle_detections)
        plate_n = sum(len(d.get('plates', [])) for d in vehicle_detections)
        stats = (f" LPR LIVE   vehicles:{veh_n}  plates:{plate_n}  "
                 f"sessions:{active}  direction:{DIRECTION}")
        cv2.putText(display, stats, (6, 24), FONT, FONT_MD,
                    (0, 255, 120), THICK_SM, cv2.LINE_AA)

        return display

    # =====================================================
    # MAIN LOOP
    # =====================================================

    def run(self):
        logging.info("[WORKER] Started — entering main frame loop")

        frame_index   = 0
        decoded_frames = 0
        fps_timer     = time.time()
        fps_count     = 0
        fps_display   = 0.0

        while True:
            # Grab frame only (no decode yet)
            grabbed = self.cap.grab()
            if not grabbed:
                logging.warning("[WORKER] Stream ended or frame grab failed — exiting loop")
                break

            frame_index += 1

            # Frame skip
            if frame_index % FRAME_SKIP != 0:
                continue

            # Decode frame only when needed
            ret, frame = self.cap.retrieve()
            if not ret:
                continue

            decoded_frames += 1
            fps_count      += 1
            original_frame  = frame

            # ---- FPS calculation (update every second) ----
            now = time.time()
            if now - fps_timer >= 1.0:
                fps_display = fps_count / (now - fps_timer)
                fps_count   = 0
                fps_timer   = now

            # Vehicle detection on FULL FRAME
            results = self.vehicle.detect(original_frame)

            vehicle_count      = 0
            plate_count        = 0
            vehicle_detections = []   # for display

            if results and len(results[0].boxes) > 0:

                vehicle_boxes = results[0].boxes[:MAX_VEHICLES_PER_FRAME]
                vehicle_count = len(vehicle_boxes)

                for box in vehicle_boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bbox = (x1, y1, x2, y2)

                    vehicle_crop = safe_crop(original_frame, bbox)
                    if vehicle_crop is None:
                        continue

                    # Plate detection
                    plates = self.plate.detect(vehicle_crop)

                    # Collect absolute plate coords for display
                    abs_plate_boxes = []
                    if plates:
                        plate_count += len(plates)
                        logging.debug(
                            f"[PLATE DETECT]  vehicle@({x1},{y1})-({x2},{y2})  "
                            f"{len(plates)} plate(s) found"
                        )
                        for px1, py1, px2, py2, _conf in plates:
                            # Convert vehicle-relative coords → full-frame coords
                            abs_plate_boxes.append((
                                x1 + px1, y1 + py1,
                                x1 + px2, y1 + py2
                            ))

                    # Match / create session
                    sid = self._match_session(bbox)
                    if sid is None:
                        sid = self._create_session(bbox)
                    self._update_session(sid, bbox)

                    # ---- Centroid line-crossing detection ----
                    session = self.sessions[sid]
                    lx1, ly1, lx2, ly2 = self.line
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Signed cross product: tells which side of the line the centroid is on
                    cross = (lx2 - lx1) * (cy - ly1) - (ly2 - ly1) * (cx - lx1)
                    curr_side = 1 if cross > 0 else (-1 if cross < 0 else 0)
                    prev_side = session["last_side"]

                    if (prev_side is not None
                            and prev_side != 0
                            and curr_side != 0
                            and prev_side != curr_side):
                        # Side changed → line crossed
                        crossed = "ENTRY" if prev_side < 0 and curr_side > 0 else "EXIT"
                        session["direction_crossed"] = crossed
                        logging.info(
                            f"[LINE CROSS]   session={sid[:8]}  direction={crossed}  "
                            f"centroid=({cx},{cy})"
                        )

                    session["last_side"] = curr_side

                    for px1, py1, px2, py2, _conf in plates:
                        plate_crop = safe_crop(vehicle_crop, (px1, py1, px2, py2))
                        if plate_crop is None:
                            continue

                        plate_crop = cv2.resize(
                            plate_crop,
                            None,
                            fx=2.0,
                            fy=2.0,
                            interpolation=cv2.INTER_CUBIC
                        )
                        self._save_plate_image(sid, plate_crop, vehicle_crop)

                    vehicle_detections.append({
                        'bbox':   bbox,
                        'sid':    sid,
                        'plates': abs_plate_boxes,
                    })

            # Log frame summary every 10 decoded frames
            if decoded_frames % 10 == 0:
                logging.info(
                    f"[FRAME {decoded_frames:05d}]  vehicles={vehicle_count}  "
                    f"plates={plate_count}  active_sessions={len(self.sessions)}"
                )

            # Cleanup expired sessions
            self._cleanup_sessions()

            # ---- Live inference window ----
            if SHOW_LPR_WINDOW:
                display_frame = self._draw_inference_frame(
                    original_frame, vehicle_detections
                )

                # FPS badge (bottom-left)
                h_d, w_d = display_frame.shape[:2]
                cv2.putText(
                    display_frame,
                    f"FPS: {fps_display:.1f}",
                    (8, h_d - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (100, 255, 100), 1, cv2.LINE_AA
                )

                win_name = "LPR Live Inference"
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_name, min(w_d, 1280), min(h_d, 720))
                cv2.imshow(win_name, display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        # Finalize remaining sessions
        for sid in list(self.sessions.keys()):
            self._finalize_session(sid)

        self.cap.release()
        cv2.destroyAllWindows()
        logging.info("[WORKER] Stopped")

