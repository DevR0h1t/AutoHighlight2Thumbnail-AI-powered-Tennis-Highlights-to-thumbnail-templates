from ultralytics import YOLO
import cv2
import pickle
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path, device='cpu', conf=0.5):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf

    def choose_and_filter_players(self, court_kpts, player_detections):
        # choose two closest players to court corners in first frame
        first = player_detections[0]
        chosen = self.choose_players(court_kpts[0], first)
        filtered = []
        for frame_dict in player_detections:
            filt = {tid: bbox for tid, bbox in frame_dict.items() if tid in chosen}
            filtered.append(filt)
        return filtered

    def choose_players(self, court_kpt, frame_dict):
        # court_kpt: list of 14 (x,y)
        # find two tracks with smallest min distance to any keypoint
        distances = []
        keypoints = [(court_kpt[i], court_kpt[i+1]) for i in range(0, len(court_kpt), 2)]
        for tid, bbox in frame_dict.items():
            center = get_center_of_bbox(bbox)
            md = min(measure_distance(center, kp) for kp in keypoints)
            distances.append((tid, md))
        distances.sort(key=lambda x: x[1])
        return [distances[0][0], distances[1][0]]

    def detect_frames(self, frames, stub_path=None):
        # supports optional pickle stub
        if stub_path:
            with open(stub_path,'rb') as f:
                return pickle.load(f)
        detections = []
        for frame in frames:
            detections.append(self.detect_frame(frame))
        if stub_path:
            with open(stub_path,'wb') as f:
                pickle.dump(detections, f)
        return detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True, device=self.device, conf=self.conf)[0]
        names = results.names
        frame_dict = {}
        for box in results.boxes:
            if names[int(box.cls)] == 'person':
                tid = int(box.id.tolist()[0])
                bbox = box.xyxy[0].tolist()
                frame_dict[tid] = bbox
        return frame_dict

    def draw_bboxes(self, frame, detections, color=(0,255,0)):
        for tid, bbox in detections.items():
            x1,y1,x2,y2 = map(int, bbox)
            cv2.putText(frame, f"Player {tid}",(x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color,2)
            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
        return frame
