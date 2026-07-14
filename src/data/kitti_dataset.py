import os
import cv2
import torch
from torch.utils.data import Dataset

from src.config.configs import IMG_SIZE, GRID_SIZE, NUM_CLASSES, CLASS_TO_IDX
from src.detection.bbox import encode_targets
from src.data.transform import random_hflip

class KITTIDataset(Dataset):

    @staticmethod
    def _resolve_split_dir(path):
        if any(name.endswith('.png') for name in os.listdir(path)):
            return path

        train_dir = os.path.join(path, 'train')
        val_dir = os.path.join(path, 'val')

        if os.path.isdir(train_dir) and any(name.endswith('.png') for name in os.listdir(train_dir)):
            return train_dir

        if os.path.isdir(val_dir) and any(name.endswith('.png') for name in os.listdir(val_dir)):
            return val_dir

        return path

    def __init__(self, image_dir, label_dir, img_size=IMG_SIZE, augment=True):
        self.image_dir = self._resolve_split_dir(image_dir)
        self.label_dir = self._resolve_split_dir(label_dir)
        self.img_size = img_size
        self.augment = augment

        self.samples = sorted(f.replace(".png", "") for f in os.listdir(self.image_dir) if f.endswith(".png"))

        if not self.samples:
            raise ValueError(f"No PNG images found in '{self.image_dir}'. Check the dataset path.")

    def __len__(self):
        return len(self.samples)
    
    def _parse_label(self, label_path, orig_w, orig_h):
        boxes, labels = [], []

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) < 5:
                    continue

                class_token = parts[0]
                cx, cy, w, h = map(float, parts[1:5])

                if class_token.isdigit():
                    class_id = int(class_token)
                    if class_id < 0 or class_id >= NUM_CLASSES:
                        continue
                elif class_token in CLASS_TO_IDX:
                    class_id = CLASS_TO_IDX[class_token]
                else:
                    continue

                boxes.append((cx, cy, w, h))
                labels.append(class_id)
            
        return boxes, labels
    
    def __getitem__(self, idx):
        name = self.samples[idx]

        image = cv2.imread(os.path.join(self.image_dir, name + '.png'))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {os.path.join(self.image_dir, name + '.png')}")

        orig_h, orig_w, _ = image.shape
        label_path = os.path.join(self.label_dir, name + '.txt')
        if os.path.exists(label_path):
            boxes, labels = self._parse_label(label_path, orig_w, orig_h)
        else:
            boxes, labels = [], []

        image = cv2.resize(image, (self.img_size, self.img_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        if self.augment:
            image, boxes_tensor = random_hflip(image, boxes_tensor)

        target = encode_targets(boxes_tensor.tolist(), labels, GRID_SIZE, NUM_CLASSES)

        return image, target, boxes_tensor, torch.tensor(labels, dtype=torch.long)

    def detection_collate(batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        targets = torch.stack([item[1] for item in batch], dim=0)

        raw_boxes = [item[2] for item in batch]
        raw_labels = [item[3] for item in batch]

        return images, targets, raw_boxes, raw_labels