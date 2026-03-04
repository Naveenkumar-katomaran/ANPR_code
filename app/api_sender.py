import requests
import logging
import threading
import time
import os
from datetime import datetime
import pytz


class APISender:
    """
    Asynchronous API sender with exponential retry.
    """

    def __init__(self, url: str, key: str, camera_id: str):
        self.url = url
        self.key = key
        self.camera_id = camera_id
        logging.info(f"[API] Initialized camera_id={camera_id}")

    def _get_ist_time(self) -> str:
        ist = pytz.timezone("Asia/Kolkata")
        return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S.%f %z")

    def send_async(self, plate: str, image_path: str | None):
        thread = threading.Thread(
            target=self._send_with_retry,
            args=(plate, image_path),
            daemon=True
        )
        thread.start()

    def _send_with_retry(self, plate: str, image_path: str | None):
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                timestamp = self._get_ist_time()

                payload = {
                    "vehicle_entries[camera_id]": self.camera_id,
                    "vehicle_entries[detected_time]": timestamp,
                    "vehicle_entries[number_plate]": plate,
                    "vehicle_entries[offline_entry]": True,
                }

                headers = {
                    "Authorization": f"API_KEY {self.key}"
                }

                files = None
                image_file = None

                if image_path and os.path.exists(image_path):
                    image_file = open(image_path, "rb")
                    files = {
                        "vehicle_entries[number_plate_image]": (
                            os.path.basename(image_path),
                            image_file,
                            "image/jpeg"
                        )
                    }

                logging.info(f"[API] Attempt {attempt+1}/{max_attempts} Plate={plate}")

                response = requests.post(
                    self.url,
                    data=payload,
                    files=files,
                    headers=headers,
                    timeout=10
                )

                if image_file:
                    image_file.close()

                if response.status_code in [200, 201]:
                    logging.info(f"[API SUCCESS] Plate={plate}")
                    return

                logging.error(
                    f"[API ERROR] Status={response.status_code} Plate={plate}"
                )

            except Exception as e:
                logging.exception(f"[API EXCEPTION] Plate={plate} Error={e}")

            sleep_time = 2 ** attempt
            logging.warning(f"[API RETRY] Sleeping {sleep_time}s")
            time.sleep(sleep_time)

        logging.critical(f"[API FAILED] Plate={plate}")
