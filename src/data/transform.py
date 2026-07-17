import random

import cv2
import numpy as np
import torch


def letterbox_image(image, image_size, fill_value=114):
    """Resize an HWC OpenCV image proportionally and pad it to a square."""
    orig_h, orig_w = image.shape[:2]
    scale = min(image_size / orig_w, image_size / orig_h)
    resized_w, resized_h = round(orig_w * scale), round(orig_h * scale)
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_left = (image_size - resized_w) // 2
    pad_top = (image_size - resized_h) // 2
    canvas = np.full((image_size, image_size, 3), fill_value, dtype=image.dtype)
    canvas[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w] = resized
    return canvas, (resized_w / orig_w, resized_h / orig_h, pad_left, pad_top)


def letterbox_boxes(boxes, orig_w, orig_h, image_size, transform):
    """Map normalized original-image boxes into normalized letterboxed space."""
    if boxes.numel() == 0:
        return boxes

    scale_x, scale_y, pad_left, pad_top = transform
    boxes = boxes.clone()
    boxes[:, 0] = (boxes[:, 0] * orig_w * scale_x + pad_left) / image_size
    boxes[:, 1] = (boxes[:, 1] * orig_h * scale_y + pad_top) / image_size
    boxes[:, 2] = boxes[:, 2] * orig_w * scale_x / image_size
    boxes[:, 3] = boxes[:, 3] * orig_h * scale_y / image_size
    return boxes

def hflip(image, boxes):
    image = image.flip(-1)

    if len(boxes) > 0:
        boxes = boxes.clone()
        boxes[:, 0] = 1.0 - boxes[:, 0]

    return image, boxes

def color_jitter(image, brightness=0.2, contrast=0.2):

    if brightness > 0:
        delta = random.uniform(-brightness, brightness)
        image = image + delta

    if contrast > 0:
        factor = random.uniform(1 - contrast, 1 + contrast)
        mean = image.mean(dim=(1, 2), keepdim=True)
        image = (image - mean) * factor + mean

    return image.clamp(0, 1)

def random_color_jitter(image, brightness=0.2, contrast=0.2, p=0.5):
    if random.random() < p:
        return color_jitter(image, brightness, contrast)

    return image

def random_crop_scale(image, boxes, labels, extra=None, scale_range=(0.8, 1.0), min_visibility=0.3, debug=False):

    _, H, W = image.shape
    scale = random.uniform(*scale_range)

    crop_h, crop_w = int(H * scale), int(W * scale)
    top = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)

    cropped = image[:, top:top + crop_h, left:left + crop_w]
    cropped = torch.nn.functional.interpolate(
        cropped.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
    ).squeeze(0)

    if len(boxes) == 0:
        return cropped, boxes, labels, extra

    cx, cy, w, h = boxes[:, 0] * W, boxes[:, 1] * H, boxes[:, 2] * W, boxes[:, 3] * H
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2

    orig_area = (x2 - x1) * (y2 - y1)

    nx1 = x1.clamp(left, left + crop_w)
    ny1 = y1.clamp(top, top + crop_h)
    nx2 = x2.clamp(left, left + crop_w)
    ny2 = y2.clamp(top, top + crop_h)

    if debug:
        was_clipped = (nx1 != x1) | (ny1 != y1) | (nx2 != x2) | (ny2 != y2)
        print(f"Clipped {was_clipped.sum().item()} boxes out of {len(boxes)}")

    new_area = (nx2 - nx1) * (ny2 - ny1)
    keep = (new_area / orig_area) >= min_visibility

    if keep.sum() == 0:
        return image, boxes, labels, extra

    nx1, ny1, nx2, ny2 = nx1[keep] - left, ny1[keep] - top, nx2[keep] - left, ny2[keep] - top

    new_cx = (nx1 + nx2) / 2 / crop_w
    new_cy = (ny1 + ny2) / 2 / crop_h
    new_w = (nx2 - nx1) / crop_w
    new_h = (ny2 - ny1) / crop_h

    new_boxes = torch.stack([new_cx, new_cy, new_w, new_h], dim=1)
    new_labels = labels[keep]
    new_extra = extra[keep] if extra is not None else None

    return cropped, new_boxes, new_labels, new_extra

def random_augment(image, boxes, labels, extra=None, p_flip=0.5, p_crop=0.5, debug=False):
    if random.random() < p_crop:
        image, boxes, labels, extra = random_crop_scale(image, boxes, labels, extra, scale_range=(0.75,1.0), debug=debug)

    if random.random() < p_flip:
        image, boxes = hflip(image, boxes)

    image = random_color_jitter(image)

    return image, boxes, labels, extra

def mosaic_augment(samples, img_size):
    half = img_size // 2
    canvas = torch.zeros((3, img_size, img_size), dtype=samples[0][0].dtype)
    quadrant_offset = [(0,0), (0, half), (half, 0), (half, half)]

    all_boxes = []
    all_labels = []

    for (image, boxes, labels), (top, left) in zip(samples, quadrant_offset):
        resized = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(half, half), mode='bilinear', align_corners=False
        ).squeeze(0)
        canvas[:, top:top + half, left:left + half] = resized

        if boxes.numel() > 0:
            cx = (boxes[:, 0] * half + left) / img_size
            cy = (boxes[:, 1] * half + top) / img_size
            w = boxes[:, 2] * half / img_size
            h = boxes[:, 3] * half / img_size
            all_boxes.append(torch.stack([cx, cy, w, h], dim=1))
            all_labels.append(labels)

    if all_boxes:
        combined_boxes = torch.cat(all_boxes, dim=0)
        combined_labels = torch.cat(all_labels, dim=0)
    else:
        combined_boxes = samples[0][1].new_zeros((0, 4))
        combined_labels = samples[0][2].new_zeros((0,), dtype=torch.long)

    return canvas, combined_boxes, combined_labels