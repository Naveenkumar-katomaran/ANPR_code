import threading
import queue
import logging
import time
from app.detection.ocr import OCR

class OCRManager:
    """
    Manages OCR processing in a background thread to prevent blocking the main loop.
    """
    def __init__(self, model_path: str, device: str):
        self.ocr = OCR(model_path, device)
        self.queue = queue.Queue(maxsize=100) # Buffer for plates
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._worker, daemon=True, name="OCRWorker")
        self.thread.start()
        logging.info("[OCR MANAGER] Initialized background worker")

    def enqueue_plate(self, session, plate_img):
        """
        Add a plate image to the queue for background processing.
        """
        try:
            # We store a reference to the session dict so we can update it directly
            self.queue.put_nowait((session, plate_img))
        except queue.Full:
            logging.warning("[OCR MANAGER] Queue full, dropping plate image")

    def _worker(self):
        logging.info("[OCR MANAGER] Worker thread started")
        batch_size = 8  # Small batch for responsiveness
        
        while self.running:
            items = []
            # Wait for at least one item
            try:
                items.append(self.queue.get(timeout=1.0))
            except queue.Empty:
                continue

            # Try to grab more for a batch without waiting too long
            try:
                while len(items) < batch_size:
                    items.append(self.queue.get_nowait())
            except queue.Empty:
                pass

            try:
                if not items: continue

                # batch processing
                images = [it[1] for it in items]
                results = self.ocr.read_batch_with_confidence(images)
                
                for i, (session, _) in enumerate(items):
                    text, conf = results[i]
                    
                    if text and conf > 0:
                        with self.lock:
                            if "candidates" not in session:
                                session["candidates"] = []
                            session["candidates"].append({"text": text, "conf": conf})
                        
                        logging.debug(f"[OCR ASYNC] session={session['id'][:8]} text={text} conf={conf:.2f}")
                    self.queue.task_done()

            except Exception as e:
                logging.error(f"[OCR MANAGER] Batch processing error: {e}")
                for _ in range(len(items)): self.queue.task_done()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
