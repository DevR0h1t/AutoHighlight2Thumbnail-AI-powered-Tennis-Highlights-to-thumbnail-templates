import json, os

class HighlightExporter:
    def __init__(self, output_dir, fps):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.fps = fps
        self.frames = []
        self.meta = []

    def add_frame(self, frame_idx, frame, shot_speed=None):
        self.frames.append(frame)
        self.meta.append({'frame':frame_idx, 'speed':shot_speed})

    def get_frames(self):
        return self.frames

    def save_metadata(self, path):
        with open(path,'w') as f:
            json.dump(self.meta, f, indent=2)