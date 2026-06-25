import cv2
import threading
from queue import Queue
import time

class AsyncStreamProcessor:
    """
    Decouples high-frequency frame capture loops from downstream 
    deep learning model inference latencies using asynchronous memory queues.
    """
    def __init__(self, source=0, max_queue_size=5):
        self.stream = cv2.VideoCapture(source)
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.stopped = False
        self.thread = None

    def start(self):
        self.stopped = False
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self):
        while not self.stopped:
            if not self.frame_queue.full():
                grabbed, frame = self.stream.read()
                if not grabbed:
                    self.stop()
                    break
                # Destructively overwrite oldest queue item if system bottleneck stalls downstream
                self.frame_queue.put(frame)
            else:
                time.sleep(0.01) # Yield execution slices dynamically

    def read_frame(self):
        """Fetches the latest synchronized matrix block from the queue."""
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None

    def stop(self):
        self.stopped = True
        if self.thread:
            self.thread.join()
        self.stream.release()
