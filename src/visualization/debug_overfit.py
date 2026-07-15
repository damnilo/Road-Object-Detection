import argparse
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.config.configs import FINE_GRID_SIZE, GRID_SIZE, NUM_CLASSES
from src.data.kitti_dataset import KITTIDataset
from src.detection.bbox import decode_multiscale_predictions, decode_targets, xywh_to_xyxy
from src.models.detector import Detector


def tensor_to_bgr(image):
    image = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def draw_xywh_boxes(frame, boxes, labels, color, prefix=''):
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
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return frame


def build_subset_dataset(sample_count, seed, augment=False):
    dataset = KITTIDataset('dataset/images/train', 'dataset/labels/train', augment=augment)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    selected_indices = indices[:min(sample_count, len(indices))]
    return dataset, Subset(dataset, selected_indices), selected_indices


@torch.no_grad()
def inspect_batch(model, dataloader, output_dir, device, conf_thresh=0.25, nms_iou_thresh=0.45):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    for batch_index, (images, targets, raw_boxes_batch, raw_labels_batch) in enumerate(dataloader):
        images = images.to(device)
        preds = model(images)

        for image_index in range(images.size(0)):
            image = tensor_to_bgr(images[image_index])
            target = {scale: value[image_index] for scale, value in targets.items()}
            raw_boxes = raw_boxes_batch[image_index]
            raw_labels = raw_labels_batch[image_index]

            decoded_targets = [decode_targets(value, num_classes=NUM_CLASSES) for value in target.values()]
            target_boxes = torch.cat([item[0] for item in decoded_targets])
            target_labels = torch.cat([item[1] for item in decoded_targets])
            pred_boxes, pred_scores, pred_labels = decode_multiscale_predictions(
                {scale: value[image_index] for scale, value in preds.items()},
                {'fine': FINE_GRID_SIZE, 'coarse': GRID_SIZE}, NUM_CLASSES,
                conf_threshold=conf_thresh,
            )

            pre_nms_count = pred_boxes.size(0)

            if pred_boxes.numel() > 0:
                from src.detection.bbox import nms

                keep = nms(pred_boxes, pred_scores, pred_labels, iou_threshold=nms_iou_thresh)
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]

            post_nms_count = pred_boxes.size(0)

            frame = draw_xywh_boxes(image.copy(), raw_boxes, raw_labels, (0, 255, 0), 'raw:')
            frame = draw_xywh_boxes(frame, target_boxes, target_labels, (0, 0, 255), 'tgt:')

            if pred_boxes.numel() > 0:
                pred_xywh = pred_boxes.clone()
                pred_xywh[:, 0] = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
                pred_xywh[:, 1] = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
                pred_xywh[:, 2] = pred_boxes[:, 2] - pred_boxes[:, 0]
                pred_xywh[:, 3] = pred_boxes[:, 3] - pred_boxes[:, 1]

                boxes_xyxy = pred_boxes.cpu().numpy()
                height, width = frame.shape[:2]
                for box, label, score in zip(boxes_xyxy, pred_labels.cpu().tolist(), pred_scores.cpu().tolist()):
                    x1 = int(round(box[0] * width))
                    y1 = int(round(box[1] * height))
                    x2 = int(round(box[2] * width))
                    y2 = int(round(box[3] * height))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        frame,
                        f'pred:{label} {score:.2f}',
                        (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

            caption = (
                f'batch={batch_index} idx={image_index} '
                f'raw={len(raw_labels)} tgt={len(target_labels)} '
                f'pred={post_nms_count} pre_nms={pre_nms_count}'
            )
            cv2.putText(frame, caption, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            output_path = os.path.join(output_dir, f'debug_{batch_index:02d}_{image_index:02d}.png')
            cv2.imwrite(output_path, frame)
            print(f'Saved {output_path}')


def main(weights_path, sample_count=16, batch_size=4, seed=42, output_dir='diagnostics/overfit_debug', device='cpu'):
    _, subset, selected_indices = build_subset_dataset(sample_count, seed, augment=False)
    print(f'Selected indices: {selected_indices}')

    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=KITTIDataset.detection_collate,
    )

    model = Detector(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    inspect_batch(model, dataloader, output_dir, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debug an overfit run by visualizing targets and predictions.')
    parser.add_argument('--weights', default='checkpoints/best_detector_weights.pth')
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='diagnostics/overfit_debug')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    main(
        weights_path=args.weights,
        sample_count=args.samples,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
    )
