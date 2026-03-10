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
# from app.detection.vehicle_detector import VehicleDetector
from app.detection.plate_detector import PlateDetector
from app.detection.ocr_manager import OCRManager
from app.duplicate_cache import DuplicateCache
from app.api_sender import APISender
from app.utils import safe_crop, open_capture
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



# removed local open_capture, now in utils.py


# =====================================================
# CAMERA WORKER
# =====================================================

class CameraWorker:

    def __init__(self):
        logging.info("[WORKER] Initializing CameraWorker")

        self.cap = open_capture(RTSP_URL)
        if not self.cap or not self.cap.isOpened():
            logging.error("[RTSP] Failed to open stream")

        device = DEVICE
        logging.info(f"[DEVICE] Using {device}")

        # Optimization: Set torch threads if running on CPU
        if device == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)
            logging.info(f"[CPU OPTIMIZATION] Set torch threads to {os.cpu_count()}")

        # self.vehicle = VehicleDetector("models/yolov8n.pt", device)
        self.plate = PlateDetector(
            None,
            "models/plate_detection/plate_detection_model_v2.pt",
            device
        )
        self.ocr_manager = OCRManager("models/ocr/best.pt", device)

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
        # load_camera_line returns native coords + the resolution they were drawn at
        x1, y1, x2, y2, native_w, native_h = load_camera_line(RTSP_URL)

        # Scale native line coords → actual capture resolution
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or native_w
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or native_h
        sx = actual_w / native_w if native_w else 1.0
        sy = actual_h / native_h if native_h else 1.0

        self.line = (
            int(x1 * sx), int(y1 * sy),
            int(x2 * sx), int(y2 * sy),
        )
        self._last_scale = 1.0  # Default scale (updated per-frame during resize)
        logging.info(
            f"[DIRECTION] mode={DIRECTION}  "
            f"native_line=({x1},{y1})-({x2},{y2}) @ {native_w}×{native_h}  "
            f"→ scaled_line=({self.line[0]},{self.line[1]})-({self.line[2]},{self.line[3]}) @ {actual_w}×{actual_h}"
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
            if iou > 0.1 and iou > best_iou:
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
        """Buffer plate crops in memory for the session.
        Stops buffering once MAX_BATCH_IMAGES is reached.
        """
        session = self.sessions.get(sid)
        if not session:
            return

        # ---- Cap check ----
        if session["frame_count"] >= MAX_BATCH_IMAGES:
            if session["frame_count"] == MAX_BATCH_IMAGES:
                logging.info(f"[CAP REACHED] session={sid[:8]} limit={MAX_BATCH_IMAGES}")
            return

        if "crops" not in session:
            session["crops"] = []
        
        session["frame_count"] += 1
        # Store in memory
        session["crops"].append({
            "plate": plate_crop,
            "vehicle": vehicle_crop,
            "timestamp": time.time()
        })

        logging.debug(f"[PLATE BUFFERED] session={sid[:8]} frame=#{session['frame_count']:04d}")

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

        # ---- Direction gate ----
        if DIRECTION != "BOTH":
            crossed = session.get("direction_crossed")
            if crossed != DIRECTION:
                reason = f"wrong_direction (crossed={crossed}, required={DIRECTION})"
                logging.info(f"[DIRECTION DISCARD] session={sid[:8]} {reason}")
                self._save_session_to_disk(session, status="discarded", discard_reason=reason)
                _cleanup()
                del self.sessions[sid]
                return

        # ---- Minimum candidates check (instead of frames) ----
        with self.ocr_manager.lock:
            candidates = session.get("candidates", [])
            # Deep copy or at least copy the list so we can work on it safely
            candidates = list(candidates)

        if len(candidates) < MIN_FRAMES_BEFORE_DECISION:
            reason = "too_few_ocr_results"
            logging.info(
                f"[SKIP] session={sid[:8]} reason={reason} "
                f"have={len(candidates)} need>={MIN_FRAMES_BEFORE_DECISION}"
            )
            self._save_session_to_disk(session, status="skipped", discard_reason=reason)
            _cleanup()
            del self.sessions[sid]
            return

        # ---- Pick best plate from accumulated candidates ----
        plate, conf = choose_best_plate(candidates)

        # Summarize results
        from collections import Counter
        text_freq = Counter(c["text"] for c in candidates)
        logging.info(
            f"[OCR SUMMARY] session={sid[:8]} readings={text_freq} "
            f"winner='{plate}' score={conf:.2f}"
        )

        if not plate:
            reason = "no_valid_plate_pattern"
            logging.info(f"[DISCARD] session={sid[:8]} reason={reason}")
            self._save_session_to_disk(session, status="discarded", discard_reason=reason)
            _cleanup()
            del self.sessions[sid]
            return

        # ---- Duplicate check ----
        if self.dup_cache.is_duplicate(plate):
            logging.info(f"[DUPLICATE] plate={plate} session={sid[:8]} → blocked")
            self._save_session_to_disk(session, status="duplicate", discard_reason="duplicate_blocked", duplicate_blocked=True)
            _cleanup()
            del self.sessions[sid]
            return

        session["best_plate"] = plate
        session["best_conf"] = conf
        logging.info(f"[✅ FINALIZED] plate={plate} score={conf:.2f} session={sid[:8]}")
        
        # Save only the best results to disk to reduce I/O
        crops = session.get("crops", [])
        plate_path = None
        if crops:
            # For now, we take the first buffered crop. In the future, we could pick sharpest.
            best_item = crops[0] 
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plate_name = f"{plate}_{timestamp}.jpg"
            plate_path = os.path.join("outputs/plates", plate_name)
            cv2.imwrite(plate_path, best_item["plate"])
            session["best_plate_image_path"] = plate_path
            
            if STORE_SESSION_PROOF and best_item["vehicle"] is not None:
                proof_name = f"proof_{sid[:8]}_{timestamp}.jpg"
                proof_path = os.path.join("outputs/proofs", proof_name)
                cv2.imwrite(proof_path, best_item["vehicle"])
                session["session_proof_path"] = proof_path

        # Send to API
        self.sender.send_async(plate, session.get("best_plate_image_path"))
        self._save_session_to_disk(session, status="finalized", plate_path=plate_path)

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

    def _draw_inference_frame(self, frame, plate_detections):
        """
        Render a rich annotated copy of the frame for the live display.

        plate_detections: list of dicts:
            {
              'bbox': (x1,y1,x2,y2),
              'sid':  session_id or None,
              'conf': float
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

        # ---- Draw each plate ----
        for det in plate_detections:
            px1, py1, px2, py2 = det['bbox']
            sid   = det.get('sid')
            conf  = det.get('conf', 0.0)

            # Plate box
            cv2.rectangle(display, (px1, py1), (px2, py2), COLOR_PLATE, THICK_MD)

            # Centroid
            cx = (px1 + px2) // 2
            cy = (py1 + py2) // 2
            cv2.circle(display, (cx, cy), 4, COLOR_CENTROID, -1)

            # Session label
            if sid and sid in self.sessions:
                session  = self.sessions[sid]
                sid_short = sid[:8]
                frames   = session['frame_count']
                best_txt = session.get('best_plate') or '—'
                dir_txt  = session.get('direction_crossed') or '--'
                label    = f"ID:{sid_short} f:{frames} dir:{dir_txt} plt:{best_txt}"
            else:
                label = f"PLATE {conf:.2f}"

            _label(display, label, px1, max(py1 - 6, 14),
                   font_scale=FONT_MD, thickness=THICK_SM,
                   bg_color=(0, 100, 150))

        # ---- Virtual crossing line + ENTRY/EXIT arrows ----
        if self.line:
            # dlx/dly are display (detector frame) coordinates.
            # self.line is in NATIVE coords.
            s = getattr(self, '_last_scale', 1.0)
            
            lx1, ly1, lx2, ly2 = self.line
            dlx1, dly1 = int(lx1 * s), int(ly1 * s)
            dlx2, dly2 = int(lx2 * s), int(ly2 * s)

            cv2.line(display, (dlx1, dly1), (dlx2, dly2), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(display, (dlx1, dly1), 6, YELLOW, -1)
            cv2.circle(display, (dlx2, dly2), 6, YELLOW, -1)

            # Compute perpendicular direction arrows
            dx, dy = dlx2 - dlx1, dly2 - dly1
            length = math.hypot(dx, dy)
            if length > 0:
                ux, uy = dx / length, dy / length
                entry_vec = (-uy, ux)    # left of P1→P2
                exit_vec  = ( uy, -ux)  # right of P1→P2
                mid = ((dlx1 + dlx2) // 2, (dly1 + dly2) // 2)
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
        plate_n = len(plate_detections)
        stats = (f" LPR LIVE   plates:{plate_n}  "
                 f"sessions:{active}  direction:{DIRECTION}")
        cv2.putText(display, stats, (6, 24), FONT, FONT_MD,
                    (0, 255, 120), THICK_SM, cv2.LINE_AA)

        return display

    # =====================================================
    # MAIN LOOP
    # =====================================================

    def run(self):
        logging.info("[WORKER] Started — entering main frame loop")

        reconnect_delay = 5          # seconds; doubles on each failure, capped at 60
        MAX_RECONNECT_DELAY = 60

        while True:                  # ← outer reconnect loop
            frame_index    = 0
            decoded_frames = 0
            fps_timer      = time.time()
            fps_count      = 0
            fps_display    = 0.0
            user_quit      = False

            # ── inner frame loop ──────────────────────────────────────
            while True:
                # Grab frame only (no decode yet)
                grabbed = self.cap.grab()
                if not grabbed:
                    logging.warning("[WORKER] Stream ended or frame grab failed — will reconnect")
                    break                # exit inner loop → reconnect

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

                # ---- Resize for detection & display efficiency ----
                h_orig, w_orig = original_frame.shape[:2]
                scale = 1.0
                if w_orig > RESIZE_WIDTH:
                    scale = RESIZE_WIDTH / w_orig
                    detector_frame = cv2.resize(original_frame, (RESIZE_WIDTH, int(h_orig * scale)))
                else:
                    detector_frame = original_frame
                
                self._last_scale = scale # Store for visualization

                # Plate detection directly on DETECTOR FRAME (Single Stage)
                plates_found = self.plate.detect(detector_frame)

                plate_count        = 0
                plate_detections   = []   # for display

                if plates_found:
                    for px1, py1, px2, py2, conf in plates_found:
                        # Scale back to original for high-res crop
                        x1 = int(px1 / scale)
                        y1 = int(py1 / scale)
                        x2 = int(px2 / scale)
                        y2 = int(py2 / scale)
                        bbox_orig = (x1, y1, x2, y2)

                        # Match / create session using Plate Bbox
                        sid = self._match_session(bbox_orig)
                        if not sid:
                            sid = self._create_session(bbox_orig)
                        else:
                            self._update_session(sid, bbox_orig)

                        session = self.sessions[sid]

                        # ---- Centroid line-crossing detection (Plate centroid) ----
                        lx1, ly1, lx2, ly2 = self.line
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        cross = (lx2 - lx1) * (cy - ly1) - (ly2 - ly1) * (cx - lx1)
                        curr_side = 1 if cross > 0 else (-1 if cross < 0 else 0)
                        prev_side = session.get("last_side")

                        if (prev_side is not None and prev_side != 0 and curr_side != 0 and prev_side != curr_side):
                            crossed = "ENTRY" if prev_side < 0 and curr_side > 0 else "EXIT"
                            if session.get("direction_crossed") is None:
                                session["direction_crossed"] = crossed
                                logging.info(f"[LINE CROSS] session={sid[:8]} plate_centroid=({cx},{cy}) dir={crossed}")
                        
                        session["last_side"] = curr_side

                        with self.ocr_manager.lock:
                            candidates_count = len(session.get("candidates", []))
                        
                        if candidates_count < 20:
                            plate_crop = safe_crop(original_frame, bbox_orig)
                            if plate_crop is not None:
                                # Higher resolution for OCR
                                plate_crop_ocr = cv2.resize(plate_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                                
                                # Context crop for "proof" (Expand plate box to show surroundings)
                                h_orig, w_orig = original_frame.shape[:2]
                                pad_h = int((y2 - y1) * 2.5) # pad around plate
                                pad_w = int((x2 - x1) * 1.5)
                                context_box = (
                                    max(0, x1 - pad_w), max(0, y1 - pad_h),
                                    min(w_orig, x2 + pad_w), min(h_orig, y2 + pad_h)
                                )
                                context_crop = safe_crop(original_frame, context_box)

                                # Buffer local crops
                                self._save_plate_image(sid, plate_crop, context_crop)
                                self.ocr_manager.enqueue_plate(session, plate_crop_ocr)
                        
                        plate_count += 1
                        plate_detections.append({
                            'bbox': (px1, py1, px2, py2), # detector frame coords
                            'sid': sid,
                            'conf': conf
                        })

                # Log frame summary every 10 decoded frames
                if decoded_frames % 10 == 0:
                    logging.info(
                        f"[FRAME {decoded_frames:05d}]  plates={plate_count}  "
                        f"active_sessions={len(self.sessions)}"
                    )

                # Cleanup expired sessions
                self._cleanup_sessions()

                # ---- Live inference window ----
                if SHOW_LPR_WINDOW:
                    display_frame = self._draw_inference_frame(
                        detector_frame, plate_detections
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
                        user_quit = True
                        break          # user pressed Q → clean shutdown

            # ── inner loop exited ──────────────────────────────────────

            # Release the dead capture handle
            try:
                self.cap.release()
            except Exception:
                pass

            if user_quit:
                logging.info("[WORKER] User quit — shutting down")
                break                  # exit outer loop → shut down

            # Finalize any active sessions before reconnecting so data is not lost
            logging.info("[WORKER] Finalizing active sessions before reconnect …")
            for sid in list(self.sessions.keys()):
                self._finalize_session(sid)

            # ── reconnect with exponential backoff ────────────────────
            logging.warning(
                f"[WORKER] Stream lost — reconnecting in {reconnect_delay}s …"
            )
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, MAX_RECONNECT_DELAY)

            new_cap = open_capture(RTSP_URL)
            if new_cap and new_cap.isOpened():
                self.cap = new_cap
                reconnect_delay = 5    # reset backoff on successful connect
                logging.info("[WORKER] Reconnected — resuming frame loop")
            else:
                logging.error(
                    f"[WORKER] Reconnect failed — will retry in {reconnect_delay}s"
                )
                # self.cap is already released; loop will try again

        # ── final shutdown ─────────────────────────────────────────────
        for sid in list(self.sessions.keys()):
            self._finalize_session(sid)

        self.ocr_manager.stop()        # Shut down background thread
        cv2.destroyAllWindows()
        logging.info("[WORKER] Stopped")

