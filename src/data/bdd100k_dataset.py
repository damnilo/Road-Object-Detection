import json
import os

import cv2
import torch
from torch.utils.data import Dataset

from src.data.transform import letterbox_image, letterbox_boxes, random_augment
from src.detection.bbox import encode_multiscale_targets
from src.config.configs import *

class BDD100KDataset(Dataset):

    def __init__(self, image_dir, label_json_path, img_size=416, augment=True,
                 debug=False, subset_names=None):
        self.image_dir = image_dir
        self.label_json_path = label_json_path
        self.augment = augment
        self.debug = debug
        self.img_size = img_size
        self._image_index = self._build_image_index(image_dir)

        if os.path.isdir(label_json_path):
            raw_labels = []
            for fname in os.listdir(label_json_path):
                if not fname.lower().endswith('.json'):
                    continue
                full = os.path.join(label_json_path, fname)
                try:
                    with open(full, 'r') as f:
                        entry = json.load(f)
                        if isinstance(entry, list):
                            raw_labels.extend(entry)
                        else:
                            raw_labels.append(entry)
                except Exception:
                    continue
        else:
            with open(label_json_path, 'r') as f:
                raw_labels = json.load(f)

        self.annotations = {}
        for entry in raw_labels:
            name = entry['name']
            boxes = []

            label_list = list(entry.get('labels') or [])
            for fram in entry.get('frames') or []:
                label_list.extend(fram.get('objects') or []) 

            for label in label_list:
                category = normalize_category(label.get('category'))
                if category not in CLASS_TO_IDX:
                    continue
                box2d = label.get('box2d')
                if box2d is None:
                    continue

                boxes.append((
                    box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2'],
                    CLASS_TO_IDX[category]
                ))

            self.annotations[name] = boxes

        self.samples = subset_names if subset_names is not None else sorted(self.annotations.keys())

        if not self.samples:
            raise ValueError("No samples found in the dataset. Please check the label JSON file and subset names.")

    def __len__(self):
        return len(self.samples)
    
    @staticmethod
    def _build_image_index(image_dir):
        index = {}
        for dirpath, _dirnames, filenames in os.walk(image_dir):
            for fname in filenames:
                full_path = os.path.join(dirpath, fname)
                if fname not in index:
                    index[fname] = full_path
                
                stem = os.path.splitext(fname)[0]
                if stem not in index:
                    index[stem] = full_path
        return index
    
    def class_counts(self, indices=None, num_classes=NUM_CLASSES):
        counts = torch.zeros(num_classes, dtype=torch.long)
        indices = range(len(self.samples)) if indices is None else indices
        for idx in indices:
            name = self.samples[idx]
            for *_xyxy, class_id in self.annotations.get(name, []):
                counts[class_id] += 1

        return counts.clamp_min(1).tolist()
    
    def __getitem__(self, idx):
        name = self.samples[idx]
        image_path = self._image_index.get(name, os.path.join(self.image_dir, name))

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        orig_h, orig_w, _ = image.shape
        entries = self.annotations.get(name, [])

        boxes, labels = [], []
        for x1, y1, x2, y2, class_id in entries:
            cx = (x1 + x2) / 2 / orig_w
            cy = (y1 + y2) / 2 / orig_h
            w = (x2 - x1) / orig_w
            h = (y2 - y1) / orig_h

            if w <= 0 or h <= 0:
                continue

            boxes.append([cx, cy, w, h])
            labels.append(class_id)
        
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        image, letterbox_transform = letterbox_image(image, self.img_size)
        boxes_tensor = letterbox_boxes(boxes_tensor, orig_w, orig_h, self.img_size, letterbox_transform)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        if self.augment:
            image, boxes_tensor, labels_tensor, _ = random_augment(
                image, boxes_tensor, labels_tensor, None, debug=self.debug
            )

        if boxes_tensor.numel() > 0:
            pixel_area = (boxes_tensor[:, 2] * self.img_size) * (boxes_tensor[:, 3] * self.img_size)
            small_object_flags = pixel_area < SMALL_OBJECT_AREA
        else:
            small_object_flags = torch.zeros((0,), dtype=torch.bool)

        target = encode_multiscale_targets(
            boxes_tensor.tolist(), labels_tensor.tolist(), GRID_SIZE, FINE_GRID_SIZE, NUM_CLASSES,
            boxes_per_cell=BOXES_PER_CELL, image_size=IMG_SIZE,
            small_object_area=SMALL_OBJECT_AREA, small_object_mask=small_object_flags.tolist(),
            fine_anchors=FINE_ANCHORS, coarse_anchors=COARSE_ANCHORS
        )

        return image, target, boxes_tensor, labels_tensor
    
    def detection_collate(batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        targets = {
            scale: torch.stack([item[1][scale] for item in batch], dim=0)
            for scale in ('fine', 'coarse')
        }

        raw_boxes = [item[2] for item in batch]
        raw_labels = [item[3] for item in batch]

        return images, targets, raw_boxes, raw_labels


def build_stratified_subset(label_json_path, target_size=12000, rare_categories=('pedestrian', 'rider', 'bicycle'),
                            rare_fraction=0.4, seed=42):
    import random

    with open(label_json_path, 'r') as f:
        raw_labels = json.load(f)

    rare_names, common_names = [], []

    for entry in raw_labels:
        label_list = list(entry.get('labels') or [])
        for fram in entry.get('frames') or []:
            label_list.extend(fram.get('objects') or [])

        categories_present = {label.get('category') for label in label_list}
        if categories_present & set(rare_categories):
            rare_names.append(entry['name'])
        else:
            common_names.append(entry['name'])

    rng = random.Random(seed)
    rng.shuffle(rare_names)
    rng.shuffle(common_names)

    rare_count = min(len(rare_names), int(target_size * rare_fraction))
    common_count = min(len(common_names), target_size - rare_count)

    return rare_names[:rare_count] + common_names[:common_count]