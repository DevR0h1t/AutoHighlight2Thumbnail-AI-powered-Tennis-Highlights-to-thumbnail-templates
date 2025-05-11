import os, json
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

class KeypointsDataset(Dataset):
    def __init__(self, root, img_folder='images', label_folder='labels', image_size=(640, 640)):
        self.img_dir = os.path.join(root, img_folder)
        self.label_dir = os.path.join(root, label_folder)
        self.image_paths = sorted([
            os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir)
            if f.lower().endswith(('jpg','jpeg','png'))
        ])
        self.label_paths = sorted([
            os.path.join(self.label_dir, f) for f in os.listdir(self.label_dir)
            if f.lower().endswith('.json')
        ])
        self.transform = transforms.Compose([
            transforms.Resize(image_size), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        orig_w, orig_h = img.size
        img_t = self.transform(img)
        with open(self.label_paths[idx],'r') as f:
            data = json.load(f)
        pts = data['keypoints']
        flat = []
        for x,y in pts:
            flat += [x/orig_w, y/orig_h]
        kpts = torch.tensor(flat, dtype=torch.float32)
        return img_t, kpts