import torch

def xywh_to_xyxy(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)
    
def iou(boxes1, boxes2):
    N, M = boxes1.size(0), boxes2.size(0)

    b1 = boxes1.unsqueeze(1).expand(N, M, 4)
    b2 = boxes2.unsqueeze(0).expand(N, M, 4)

    x1 = torch.max(b1[..., 0], b2[..., 0])
    y1 = torch.max(b1[..., 1], b2[..., 1])
    x2 = torch.min(b1[..., 2], b2[..., 2])
    y2 = torch.min(b1[..., 3], b2[..., 3])

    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    area1 = (b1[..., 2] - b1[..., 0]).clamp(0) * (b1[..., 3] - b1[..., 1]).clamp(0)
    area2 = (b2[..., 2] - b2[..., 0]).clamp(0) * (b2[..., 3] - b2[..., 1]).clamp(0)

    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-6)
    
def nms(boxes_xyxy, scores, class_ids, iou_threshold=0.45):
    keep = []

    for cls in class_ids.unique():
        cls_mask = (class_ids == cls).nonzero(as_tuple=True)[0]
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = scores[cls_mask]

        order = cls_scores.argsort(descending=True)
        cls_boxes, cls_scores, cls_mask = cls_boxes[order], cls_scores[order], cls_mask[order]

        picked = []
        active = torch.ones(cls_boxes.size(0), dtype=torch.bool)

        for i in range(cls_boxes.size(0)):
            if not active[i]:
                continue

            picked.append(cls_mask[i].item())
            if i+1 > cls_boxes.size(0) - 1:
                break

            ious = iou(cls_boxes[i:i+1], cls_boxes[i+1:])
            suppress = (ious > iou_threshold).nonzero(as_tuple=True)[0] + (i + 1)
            active[suppress] = False

        keep.extend(picked)

    return torch.tensor(keep, dtype=torch.long)
    
def encode_targets(gt_boxes, gt_labels, grid_size, num_classes):
    S = grid_size
    targets = torch.zeros((S, S, 5 + num_classes))

    for (cx, cy, w, h), label in zip(gt_boxes, gt_labels):
        i, j = int(cx * S), int(cy * S)
        i, j = min(i, S - 1), min(j, S - 1)

        if targets[j, i, 0] == 1:
            continue

        tx, ty = cx * S - i, cy * S - j

        targets[j, i, 0] = 1.0
        targets[j, i, 1] = tx
        targets[j, i, 2] = ty
        targets[j, i, 3] = w ** 0.5
        targets[j, i, 4] = h ** 0.5
        targets[j, i, 5 + label] = 1.0

    return targets
    
def decode_predictions(pred, grid_size, num_classes, conf_threshold=0.5):
    S = grid_size
    device = pred.device

    obj = pred[..., 0]
    mask = obj > conf_threshold

    if mask.sum() == 0:
        return (torch.empty(0, 4, device=device),
                torch.empty(0, device=device),
                torch.empty(0, dtype=torch.long, device=device))
        
    jj, ii = mask.nonzero(as_tuple=True)

    tx = pred[jj, ii, 1]
    ty = pred[jj, ii, 2]
    sqrt_w = pred[jj, ii, 3]
    sqrt_h = pred[jj, ii, 4]
    class_probs = pred[jj, ii, 5:5+num_classes]

    cx = (ii.float() + tx) / S
    cy = (jj.float() + ty) / S
    w = sqrt_w ** 2
    h = sqrt_h ** 2

    boxes = torch.stack([cx, cy, w, h], dim=1)
    boxes_xyxy = xywh_to_xyxy(boxes)

    class_scores, class_ids = class_probs.max(dim=1)
    scores = obj[jj, ii] * class_scores

    return boxes_xyxy, scores, class_ids