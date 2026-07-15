import os
import cv2
import torch
from torch.utils.data import Dataset

from src.config.configs import BOXES_PER_CELL, CLASS_TO_IDX, GRID_SIZE, IMG_SIZE, NUM_CLASSES
from src.detection.bbox import encode_targets
from src.data.transform import letterbox_boxes, letterbox_image, random_augment

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

    def __init__(self, image_dir, label_dir, img_size=IMG_SIZE, augment=True, debug=False):
        self.image_dir = self._resolve_split_dir(image_dir)
        self.label_dir = self._resolve_split_dir(label_dir)
        self.img_size = img_size
        self.augment = augment
        self.debug = debug

        self.samples = sorted(f.replace(".png", "") for f in os.listdir(self.image_dir) if f.endswith(".png"))

        if not self.samples:
            raise ValueError(f"No PNG images found in '{self.image_dir}'. Check the dataset path.")

    def __len__(self):
        return len(self.samples)
    
    def _parse_label(self, label_path):
        boxes, labels = [], []

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
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

                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    continue

                boxes.append((cx, cy, w, h))
                labels.append(class_id)
            
        return boxes, labels

    def class_counts(self, indices=None):
        counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
        indices = range(len(self)) if indices is None else indices
        for idx in indices:
            label_path = os.path.join(self.label_dir, self.samples[idx] + '.txt')
            if not os.path.exists(label_path):
                continue
            _, labels = self._parse_label(label_path)
            for label in labels:
                counts[label] += 1
        return counts.clamp_min(1).tolist()
    
    def __getitem__(self, idx):
        name = self.samples[idx]

        image = cv2.imread(os.path.join(self.image_dir, name + '.png'))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {os.path.join(self.image_dir, name + '.png')}")

        orig_h, orig_w, _ = image.shape
        label_path = os.path.join(self.label_dir, name + '.txt')
        if os.path.exists(label_path):
            boxes, labels = self._parse_label(label_path)
        else:
            boxes, labels = [], []

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        image, letterbox_transform = letterbox_image(image, self.img_size)
        boxes_tensor = letterbox_boxes(
            boxes_tensor, orig_w, orig_h, self.img_size, letterbox_transform
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        labels_tensor = torch.tensor(labels, dtype=torch.long)
        if self.augment:
            image, boxes_tensor, labels_tensor = random_augment(image, boxes_tensor, labels_tensor, debug=self.debug)

        target = encode_targets(
            boxes_tensor.tolist(), labels_tensor.tolist(), GRID_SIZE, NUM_CLASSES, BOXES_PER_CELL
        )

        return image, target, boxes_tensor, labels_tensor

    def detection_collate(batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        targets = torch.stack([item[1] for item in batch], dim=0)

        raw_boxes = [item[2] for item in batch]
        raw_labels = [item[3] for item in batch]

        return images, targets, raw_boxes, raw_labels
