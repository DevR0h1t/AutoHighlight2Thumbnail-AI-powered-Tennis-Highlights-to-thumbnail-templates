from ultralytics import YOLO

class Detector:
    def __init__(self, model_path: str = 'yolov8n.pt', conf_thresh: float = 0.5):
        self.model = YOLO(model_path)
        self.model.conf = conf_thresh

    def detect(self, frame):
        results = self.model(frame)
        detections = []
        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                detections.append({
                    'bbox': box,        # [x1, y1, x2, y2]
                    'class': int(cls),   # 0=person, 32=ball, etc.
                    'confidence': float(conf)
                })
        return detections