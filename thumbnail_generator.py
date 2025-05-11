import cv2
import os

class ThumbnailGenerator:
    def __init__(self, output_dir: str = 'outputs/thumbnails'):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def save(self, frame, timestamp: float) -> str:
        fname = f'thumb_{timestamp:.1f}.jpg'
        path = os.path.join(self.output_dir, fname)
        cv2.imwrite(path, frame)
        return path