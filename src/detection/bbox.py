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
    
def agnostic_nms(boxes_xyxy, scores, iou_threshold=0.45):
    if boxes_xyxy.size(0) == 0:
        return torch.empty(0, dtype=torch.long, device=boxes_xyxy.device)

    order = scores.argsort(descending=True)
    boxes_xyxy = boxes_xyxy[order]

    keep = []
    active = torch.ones(boxes_xyxy.size(0), dtype=torch.bool, device=boxes_xyxy.device)

    for i in range(boxes_xyxy.size(0)):
        if not active[i]:
            continue

        keep.append(order[i].item())

        if i + 1 > boxes_xyxy.size(0) - 1:
            break

        ious = iou(boxes_xyxy[i:i+1], boxes_xyxy[i + 1:])
        suppress = (ious > iou_threshold).nonzero(as_tuple=True)[0] + (i+1)
        active[suppress] = False

    return torch.tensor(keep, dtype=torch.long, device=boxes_xyxy.device)

def nms(boxes_xyxy, scores, class_ids, iou_threshold=0.45):
    keep = []

    for cls in class_ids.unique():
        cls_mask = (class_ids == cls).nonzero(as_tuple=True)[0]
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = scores[cls_mask]

        order = cls_scores.argsort(descending=True)
        cls_boxes, cls_scores, cls_mask = cls_boxes[order], cls_scores[order], cls_mask[order]

        picked = []
        active = torch.ones(cls_boxes.size(0), dtype=torch.bool, device=cls_boxes.device)

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

    if not keep:
        return torch.empty(0, dtype=torch.long, device=boxes_xyxy.device)

    keep = torch.tensor(keep, dtype=torch.long, device=boxes_xyxy.device)
    return keep[scores[keep].argsort(descending=True)]

def anchor_iou(w, h, anchors):
    inter_w = torch.min(torch.full_like(anchors[:, 0], w), anchors[:, 0])
    inter_h = torch.min(torch.full_like(anchors[:, 1], h), anchors[:, 1])
    inter = inter_w * inter_h
    box_area = w * h
    anchor_area = anchors[:, 0] * anchors[:, 1]
    union = box_area + anchor_area - inter
    return inter / union.clamp(min=1e-6)
    
def encode_targets(gt_boxes, gt_labels, grid_size, num_classes, boxes_per_cell=1, anchors=None):
    S = grid_size
    targets = torch.zeros((S, S, boxes_per_cell, 5 + num_classes))

    if anchors is not None and not torch.is_tensor(anchors):
        anchors = torch.tensor(anchors, dtype=torch.float32)

    for (cx, cy, w, h), label in zip(gt_boxes, gt_labels):
        if not (0 <= label < num_classes and w > 0 and h > 0):
            continue
        cx, cy = min(max(cx, 0.0), 1.0 - 1e-6), min(max(cy, 0.0), 1.0 - 1e-6)
        w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
        i, j = int(cx * S), int(cy * S)
        i, j = min(i, S - 1), min(j, S - 1)

        if anchors is not None and anchors.size(0) == boxes_per_cell:
            ious = anchor_iou(w, h, anchors)
            slot = ious.argmax().item()
            occupied = targets[j, i, slot, 0].item() > 0

            if occupied:
                existing_w = targets[j, i, slot, 3].item() ** 2
                existing_h = targets[j, i, slot, 4].item() ** 2
                existing_iou = anchor_iou(existing_w, existing_h, anchors[slot:slot+1]).item()

                if ious[slot].item() <= existing_iou:
                    continue
        else:
            occupied_mask = targets[j, i, :, 0].bool()

            if (~occupied_mask).any():
                slot = (~occupied_mask).nonzero(as_tuple=True)[0][0].item()
            else:
                existing_areas = targets[j, i, :, 3].square() * targets[j, i, :, 4].square()
                slot = existing_areas.argmin().item()

                if w * h <= existing_areas[slot].item():
                    continue            

        tx, ty = cx * S - i, cy * S - j

        targets[j, i, slot, 0] = 1.0
        targets[j, i, slot, 1] = tx
        targets[j, i, slot, 2] = ty
        targets[j, i, slot, 3] = w ** 0.5
        targets[j, i, slot, 4] = h ** 0.5
        targets[j, i, slot, 5 + label] = 1.0

    return targets


def encode_multiscale_targets(gt_boxes, gt_labels, coarse_grid_size, fine_grid_size,
                              num_classes, boxes_per_cell=1, image_size=416,
                              small_object_area=32 * 32, small_object_mask=None,
                              fine_anchors=None, coarse_anchors=None):
    fine_boxes, fine_labels = [], []
    coarse_boxes, coarse_labels = [], []

    if small_object_mask is None:
        small_object_mask = [(w * image_size) * (h * image_size) < small_object_area for _, _, w, h in gt_boxes]

    for box, label, is_small in zip(gt_boxes, gt_labels, small_object_mask):
        if is_small:
            fine_boxes.append(box)
            fine_labels.append(label)
        else:
            coarse_boxes.append(box)
            coarse_labels.append(label)

    return {
        "fine": encode_targets(fine_boxes, fine_labels, fine_grid_size, num_classes, boxes_per_cell, anchors=fine_anchors),
        "coarse": encode_targets(coarse_boxes, coarse_labels, coarse_grid_size, num_classes, boxes_per_cell, anchors=coarse_anchors),
    }

def decode_targets(target, num_classes=None):
    if target.dim() != 4:
        raise ValueError('target must have shape (S, S, B, 5 + num_classes)')

    S = target.size(0)
    boxes = []
    labels = []
    cells = []

    if num_classes is None:
        num_classes = target.size(-1) - 5

    for j in range(S):
        for i in range(S):
            for box_idx in range(target.size(2)):
                cell = target[j, i, box_idx]
                if cell[0].item() <= 0:
                    continue

                tx, ty, sqrt_w, sqrt_h = cell[1:5]
                class_slice = cell[5:5 + num_classes]

                cx = (i + tx.item()) / S
                cy = (j + ty.item()) / S
                w = float(sqrt_w.item()) ** 2
                h = float(sqrt_h.item()) ** 2
                label = int(class_slice.argmax().item()) if class_slice.numel() > 0 else -1

                boxes.append([cx, cy, w, h])
                labels.append(label)
                cells.append((j, i, box_idx))

    if boxes:
        boxes = torch.tensor(boxes, dtype=target.dtype, device=target.device)
        labels = torch.tensor(labels, dtype=torch.long, device=target.device)
    else:
        boxes = target.new_zeros((0, 4))
        labels = torch.empty((0,), dtype=torch.long, device=target.device)

    return boxes, labels, cells
    
def decode_predictions(pred, grid_size, num_classes, conf_threshold=0.5):
    S = grid_size
    device = pred.device

    obj = pred[..., 0].sigmoid()
    class_scores, class_ids = pred[..., 5:5 + num_classes].softmax(dim=-1).max(dim=-1)
    scores = obj * class_scores
    mask = scores > conf_threshold

    if mask.sum() == 0:
        return (torch.empty(0, 4, device=device),
                torch.empty(0, device=device),
                torch.empty(0, dtype=torch.long, device=device))
        
    if pred.dim() == 3:
        pred = pred.unsqueeze(2)
        scores = scores.unsqueeze(2)
        class_ids = class_ids.unsqueeze(2)
        mask = mask.unsqueeze(2)

    jj, ii, box_idx = mask.nonzero(as_tuple=True)

    tx = pred[jj, ii, box_idx, 1]
    ty = pred[jj, ii, box_idx, 2]
    sqrt_w = pred[jj, ii, box_idx, 3]
    sqrt_h = pred[jj, ii, box_idx, 4]

    cx = (ii.float() + tx) / S
    cy = (jj.float() + ty) / S
    w = sqrt_w ** 2
    h = sqrt_h ** 2

    boxes = torch.stack([cx, cy, w, h], dim=1)
    boxes_xyxy = xywh_to_xyxy(boxes).clamp(0.0, 1.0)
    scores = scores[jj, ii, box_idx]
    class_ids = class_ids[jj, ii, box_idx]

    return boxes_xyxy, scores, class_ids


def decode_multiscale_predictions(predictions, grid_sizes, num_classes, conf_threshold=0.5):
    """Decode all heads into one normalized-coordinate prediction set."""
    boxes, scores, labels = [], [], []
    for scale_name, pred in predictions.items():
        scale_boxes, scale_scores, scale_labels = decode_predictions(
            pred, grid_sizes[scale_name], num_classes, conf_threshold
        )
        if scale_boxes.numel() > 0:
            boxes.append(scale_boxes)
            scores.append(scale_scores)
            labels.append(scale_labels)

    if boxes:
        return torch.cat(boxes), torch.cat(scores), torch.cat(labels)

    device = next(iter(predictions.values())).device
    return (torch.empty(0, 4, device=device), torch.empty(0, device=device),
            torch.empty(0, dtype=torch.long, device=device))
