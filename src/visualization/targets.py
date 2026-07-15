import argparse
import os
import random

import cv2
import numpy as np
import torch

from src.config.configs import NUM_CLASSES
from src.data.kitti_dataset import KITTIDataset
from src.detection.bbox import decode_targets, xywh_to_xyxy


def tensor_to_bgr(image):
    image = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def draw_xywh_boxes(frame, boxes, labels, color, prefix):
    height, width = frame.shape[:2]

    if boxes.numel() == 0:
        return frame

    boxes_xyxy = xywh_to_xyxy(boxes.cpu()).numpy()
    labels = labels.cpu().tolist()

    for box, label in zip(boxes_xyxy, labels):
        x1 = int(round(box[0] * width))
        y1 = int(round(box[1] * height))
        x2 = int(round(box[2] * width))
        y2 = int(round(box[3] * height))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f'{prefix}{label}',
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return frame


def save_augmented_target_visualizations(image_dir='dataset/images/train', label_dir='dataset/labels/train',
                                         output_dir='diagnostics/targets', sample_count=8, seed=42,
                                         augment=True):
    os.makedirs(output_dir, exist_ok=True)
    dataset = KITTIDataset(image_dir, label_dir, augment=augment, debug=True)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    for output_index, dataset_index in enumerate(indices[:min(sample_count, len(indices))]):
        image, targets, raw_boxes, raw_labels = dataset[dataset_index]
        decoded = [decode_targets(target, num_classes=NUM_CLASSES) for target in targets.values()]
        decoded_boxes = torch.cat([item[0] for item in decoded])
        decoded_labels = torch.cat([item[1] for item in decoded])

        frame = tensor_to_bgr(image)
        frame = draw_xywh_boxes(frame, raw_boxes, raw_labels, (0, 255, 0), 'raw:')
        frame = draw_xywh_boxes(frame, decoded_boxes, decoded_labels, (0, 0, 255), 'enc:')

        caption = f'idx={dataset_index} raw={len(raw_labels)} encoded={len(decoded_labels)}'
        cv2.putText(frame, caption, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        output_path = os.path.join(output_dir, f'augmented_target_{output_index:02d}.png')
        cv2.imwrite(output_path, frame)
        print(f'Saved {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize encoded targets after augmentation.')
    parser.add_argument('--image-dir', default='dataset/images/train')
    parser.add_argument('--label-dir', default='dataset/labels/train')
    parser.add_argument('--output-dir', default='diagnostics/targets')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-augment', action='store_true')
    args = parser.parse_args()

    save_augmented_target_visualizations(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        sample_count=args.count,
        seed=args.seed,
        augment=not args.no_augment,
    )
