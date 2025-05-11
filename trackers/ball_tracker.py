from ultralytics import YOLO
import cv2
import pandas as pd
import pickle
import numpy as np

class BallTracker:
    def __init__(self, model_path, device='cpu', conf=0.5, interpolate=True):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.interpolate = interpolate

    def interpolate_ball_positions(self, centers):
        """
        Fill missing ball center detections via linear interpolation and backfill.
        Input: list of (x,y) or None for each frame.
        Output: list of (x,y) floats for each frame.
        """
        # Prepare DataFrame
        pts = [(c if c is not None else (np.nan, np.nan)) for c in centers]
        df = pd.DataFrame(pts, columns=['x','y'])
        # Interpolate then backfill any remaining NaNs
        df[['x','y']] = df[['x','y']].interpolate().bfill()
        # Return list of tuples
        return [tuple(pt) for pt in df[['x','y']].to_numpy()]

    def get_ball_shot_frames(self, centers, window=5):
        """
        Identify frames where the ball changes vertical motion direction.
        Returns a list of frame indices.
        """
        pts = [(c if c is not None else (np.nan, np.nan)) for c in centers]
        df = pd.DataFrame(pts, columns=['x','y'])
        # smooth the y-coordinate
        df['y_smooth'] = df['y'].rolling(window=window, min_periods=1).mean()
        df['delta'] = df['y_smooth'].diff()
        shot_frames = []
        # look for sign changes in delta
        for i in range(1, len(df)-1):
            d0 = df['delta'].iloc[i]
            d1 = df['delta'].iloc[i+1]
            if not np.isnan(d0) and not np.isnan(d1) and d0 * d1 < 0:
                shot_frames.append(i)
        return shot_frames

    def detect_frames(self, frames):
        """
        Simple YOLO-based detection of ball centers per frame.
        """
        centers = []
        for frame in frames:
            results = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)
            best, best_conf = None, 0
            for r in results:
                for b in r.boxes:
                    c = float(b.conf[0])
                    if c > best_conf:
                        best_conf = c
                        x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                        best = ((x1+x2)//2, (y1+y2)//2)
            centers.append(best)
        return centers

    def draw_bboxes(self, frame, centers, color=(0,0,255)):
        for c in centers:
            if c:
                cv2.circle(frame, c, 5, color, -1)
        return frame