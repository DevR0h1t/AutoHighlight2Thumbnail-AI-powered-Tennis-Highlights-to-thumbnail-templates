import torch, cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from models.court_keypoint_model import CourtKeypointModel

class CourtLineDetector:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = CourtKeypointModel(pretrained=False).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state); self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((640,640)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def predict_frames(self, frames):
        all_kpts = []
        for frame in frames:
            h,w = frame.shape[:2]
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img)
            inp = self.transform(pil).unsqueeze(0).to(self.device)
            with torch.no_grad(): out = self.model(inp)
            pts = out[0].cpu().numpy().reshape(-1,2)
            pts[:,0] *= w; pts[:,1] *= h
            all_kpts.append(pts.tolist())
        return all_kpts

    def draw_keypoints(self, frame, kpts, color=(255,0,0)):
        for x,y in kpts:
            cv2.circle(frame, (int(x),int(y)), 4, color, -1)
        return frame