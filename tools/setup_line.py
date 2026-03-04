"""
setup_line.py — Virtual Line Setup Tool
========================================
Run once per camera to configure the crossing line.

Usage:
    python tools/setup_line.py                    # reads RTSP_URL from .env
    python tools/setup_line.py --source <url>     # override source

Controls (after the window opens):
    Left-click × 2  → set P1 then P2
    R               → reset / redraw
    S               → save to config/cameras.json
    Q               → quit without saving
"""

import cv2
import json
import os
import sys
import argparse
import math

# ── Allow running from the project root ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ── Config ────────────────────────────────────────────────────────────────────
CAMERAS_JSON = os.path.join(PROJECT_ROOT, "config", "cameras.json")
MAX_DISPLAY_WIDTH  = 1280
MAX_DISPLAY_HEIGHT = 720

# ── Colours ───────────────────────────────────────────────────────────────────
WHITE   = (255, 255, 255)
GREEN   = (0, 220,  60)
RED     = (0,  60, 220)
YELLOW  = (0, 220, 220)
BLACK   = (0,   0,   0)
CYAN    = (220, 200,  0)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scale_factors(native_w, native_h):
    """Return (display_w, display_h, scale_x, scale_y)."""
    scale = min(MAX_DISPLAY_WIDTH / native_w, MAX_DISPLAY_HEIGHT / native_h, 1.0)
    dw = int(native_w * scale)
    dh = int(native_h * scale)
    return dw, dh, native_w / dw, native_h / dh


def _draw_arrow(img, origin, direction_vec, length, color, thickness=2, tip_frac=0.25):
    """Draw an arrow from *origin* in the given unit direction."""
    ex = int(origin[0] + direction_vec[0] * length)
    ey = int(origin[1] + direction_vec[1] * length)
    cv2.arrowedLine(img, origin, (ex, ey), color, thickness,
                    tipLength=tip_frac, line_type=cv2.LINE_AA)


def _compute_perp(x1, y1, x2, y2):
    """
    Return unit vectors for ENTRY (left of P1→P2) and EXIT (right),
    and the mid-point of the line.
    """
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return None, None, None
    # Unit vec along line
    ux, uy = dx / length, dy / length
    # Left perpendicular  (ENTRY side: cross-product > 0)
    entry_vec = (-uy,  ux)
    # Right perpendicular (EXIT side)
    exit_vec  = ( uy, -ux)
    mid = ((x1 + x2) // 2, (y1 + y2) // 2)
    return entry_vec, exit_vec, mid


def _render(display_frame, points, scale_x, scale_y):
    """
    Rebuild the annotation overlay on *display_frame* (a fresh copy each call).
    points: list of (native_x, native_y) — 0, 1 or 2 items.
    Returns the annotated frame.
    """
    vis = display_frame.copy()

    # Instructions bar
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (20, 20, 20), cv2.FILLED)
    hint = ("Click P1 ..." if len(points) == 0 else
            "Click P2 ..." if len(points) == 1 else
            "S=Save  R=Reset  Q=Quit")
    cv2.putText(vis, hint, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1, cv2.LINE_AA)

    if len(points) == 0:
        return vis

    # P1 dot
    dp1 = (int(points[0][0] / scale_x), int(points[0][1] / scale_y))
    cv2.circle(vis, dp1, 7, YELLOW, -1)
    cv2.putText(vis, "P1", (dp1[0] + 9, dp1[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1, cv2.LINE_AA)

    if len(points) < 2:
        return vis

    # P2 dot
    dp2 = (int(points[1][0] / scale_x), int(points[1][1] / scale_y))
    cv2.circle(vis, dp2, 7, YELLOW, -1)
    cv2.putText(vis, "P2", (dp2[0] + 9, dp2[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1, cv2.LINE_AA)

    # Virtual line
    cv2.line(vis, dp1, dp2, WHITE, 2, cv2.LINE_AA)

    # ENTRY / EXIT arrows (in display coords)
    dx1, dy1 = int(points[0][0] / scale_x), int(points[0][1] / scale_y)
    dx2, dy2 = int(points[1][0] / scale_x), int(points[1][1] / scale_y)
    entry_vec, exit_vec, dmid = _compute_perp(dx1, dy1, dx2, dy2)

    if entry_vec is not None:
        arrow_len = 70
        _draw_arrow(vis, dmid, entry_vec, arrow_len, GREEN, thickness=2)
        _draw_arrow(vis, dmid, exit_vec,  arrow_len, RED,   thickness=2)

        ex = int(dmid[0] + entry_vec[0] * (arrow_len + 12))
        ey = int(dmid[1] + entry_vec[1] * (arrow_len + 12))
        xx = int(dmid[0] + exit_vec[0]  * (arrow_len + 12))
        xy = int(dmid[1] + exit_vec[1]  * (arrow_len + 12))

        # Label backgrounds + text
        for (lx, ly, label, color) in [
            (ex, ey, "ENTRY", GREEN),
            (xx, xy,  "EXIT",  RED),
        ]:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 4),
                          BLACK, cv2.FILLED)
            cv2.putText(vis, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return vis


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Virtual line setup tool")
    parser.add_argument("--source", default=None,
                        help="Camera source (RTSP URL or video path). "
                             "Defaults to RTSP_URL in .env")
    args = parser.parse_args()

    source = args.source or os.getenv("RTSP_URL")
    if not source:
        print("[ERROR] No source provided. Set RTSP_URL in .env or use --source.")
        sys.exit(1)

    print(f"[SETUP] Opening source: {source}")

    # ── Open capture ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        # Try integer (webcam)
        try:
            cap = cv2.VideoCapture(int(source))
        except ValueError:
            pass
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    # Skip a few frames so the stream settles (especially RTSP)
    for _ in range(10):
        cap.grab()

    ret, native_frame = cap.read()
    cap.release()

    if not ret or native_frame is None:
        print("[ERROR] Failed to read a frame from the source.")
        sys.exit(1)

    native_h, native_w = native_frame.shape[:2]
    print(f"[SETUP] Native resolution: {native_w}×{native_h}")

    disp_w, disp_h, scale_x, scale_y = _scale_factors(native_w, native_h)
    display_base = cv2.resize(native_frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    print(f"[SETUP] Display resolution: {disp_w}×{disp_h}  "
          f"(scale {scale_x:.3f}×{scale_y:.3f})")

    # ── Interactive loop ──────────────────────────────────────────────────────
    WIN = "LPR Line Setup  [click P1 then P2 | S=Save | R=Reset | Q=Quit]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)

    points = []   # list of (native_x, native_y)

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            nx = int(mx * scale_x)
            ny = int(my * scale_y)
            points.append((nx, ny))
            print(f"[SETUP] P{len(points)} set → display({mx},{my})  native({nx},{ny})")

    cv2.setMouseCallback(WIN, on_mouse)

    print("[SETUP] Window open. Click P1 then P2 on the frame.")

    while True:
        vis = _render(display_base, points, scale_x, scale_y)
        cv2.imshow(WIN, vis)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('r') or key == ord('R'):
            points.clear()
            print("[SETUP] Reset — click P1 again.")

        elif key == ord('s') or key == ord('S'):
            if len(points) < 2:
                print("[SETUP] Need both P1 and P2 before saving. Click them first.")
                continue

            x1, y1 = points[0]
            x2, y2 = points[1]

            # Load existing JSON (or start fresh)
            os.makedirs(os.path.dirname(CAMERAS_JSON), exist_ok=True)
            data = {}
            if os.path.exists(CAMERAS_JSON):
                with open(CAMERAS_JSON) as f:
                    data = json.load(f)

            data[source] = {
                "line": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "frame_width":  native_w,
                "frame_height": native_h,
            }

            with open(CAMERAS_JSON, "w") as f:
                json.dump(data, f, indent=4)

            print(f"[SETUP] ✅ Saved to {CAMERAS_JSON}")
            print(f"         source : {source}")
            print(f"         P1     : ({x1}, {y1})")
            print(f"         P2     : ({x2}, {y2})")
            break

        elif key == ord('q') or key == ord('Q'):
            print("[SETUP] Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
