from app.utils import sharpness_score
import logging

class TrackManager:
    """
    TrackManager keeps track of vehicles across frames using track IDs.
    Collects all plate crops per track for N stable frames.
    Finalizes the track by selecting the best plate using OCR confidence + sharpness.
    """

    def __init__(self, stable_count: int, max_crops: int):
        """
        Initialize TrackManager.

        Args:
            stable_count (int): Number of frames after which a track is considered stable.
            max_crops (int): Maximum number of plate crops to store per track.
        """
        self.stable_count = stable_count
        self.max_crops = max_crops
        self.tracks = {}

    def update(self, track_id: int, plate_crop):
        """
        Add a new plate crop to the track.

        Args:
            track_id (int): Unique ID of the tracked vehicle.
            plate_crop (np.array): Cropped image of the license plate.

        Returns:
            bool: True if the track has reached stable_count frames.
        """
        if track_id not in self.tracks:
            self.tracks[track_id] = {"frames": 0, "crops": []}

        track = self.tracks[track_id]
        track["frames"] += 1

        if len(track["crops"]) < self.max_crops:
            track["crops"].append(plate_crop)
            logging.debug(f"[TRACK] Added crop for track {track_id} (total {len(track['crops'])})")

        return track["frames"] >= self.stable_count

    def finalize(self, track_id: int, ocr_reader=None):
        """
        Finalize the track: choose the best plate crop.

        If OCR reader is provided, performs OCR on all crops and selects
        the plate with highest confidence combined with sharpness score.

        Args:
            track_id (int): Unique ID of the tracked vehicle.
            ocr_reader (OCR): OCR class instance with `read_with_confidence()`.

        Returns:
            tuple: (best_crop, best_text)
                - best_crop: np.array of best plate crop
                - best_text: str plate text (None if OCR not provided)
        """
        track = self.tracks.pop(track_id, None)
        if not track:
            logging.warning(f"[TRACK] No track found for {track_id}")
            return None, None

        crops = track["crops"]
        if not crops:
            logging.warning(f"[TRACK] No crops stored for track {track_id}")
            return None, None

        if ocr_reader:
            best_text = ""
            best_crop = None
            best_score = -1

            for crop in crops:
                text, conf = ocr_reader.read_with_confidence(crop)
                score = conf + sharpness_score(crop)  # combine OCR confidence + sharpness
                if score > best_score:
                    best_score = score
                    best_text = text
                    best_crop = crop

            logging.info(f"[TRACK] Finalized track {track_id}, best plate: {best_text}")
            return best_crop, best_text

        # Fallback: pick the sharpest crop
        best_crop = max(crops, key=lambda x: sharpness_score(x))
        return best_crop, None
