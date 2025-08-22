import os
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

COLORS = {
    "red": (255,0,0),
    "green": (0,255,0),
    "blue": (0,0,255),
    "yellow": (255,255,0),
    "cyan": (0,255,255),
    "magenta": (255,0,255)
}

class PolygonDataset(Dataset):
    def __init__(self, size=200, img_size=128):
        self.size = size
        self.img_size = img_size
        self.samples = [self._generate() for _ in range(size)]

    def _generate(self):
        img = Image.new("RGB", (self.img_size, self.img_size), "black")
        mask = Image.new("RGB", (self.img_size, self.img_size), "black")

        draw = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)

        num_points = np.random.randint(3, 6)
        points = [(np.random.randint(20, self.img_size-20),
                   np.random.randint(20, self.img_size-20)) for _ in range(num_points)]

        color_name, color = list(COLORS.items())[np.random.randint(len(COLORS))]

        draw.polygon(points, outline="white", fill="white")  # input (shape only)
        draw_mask.polygon(points, outline=color, fill=color)  # target (colored)

        return img, mask

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img, mask = self.samples[idx]
        img = np.array(img).transpose(2,0,1) / 255.0
        mask = np.array(mask).transpose(2,0,1) / 255.0
        return img.astype(np.float32), mask.astype(np.float32)
