import math

import torch.nn as nn
import torch
import torch.nn.functional as F

from src.config.configs import NUM_CLASSES

def ciou_loss(pred_xyxy, target_xyxy, eps=1e-7):
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    tx1, ty1, tx2, ty2 = target_xyxy.unbind(-1)

    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    p_area = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    t_area = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
    union = p_area + t_area - inter + eps
    iou = inter / union

    ex1 = torch.min(px1, tx1)
    ey1 = torch.min(py1, ty1)
    ex2 = torch.max(px2, tx2)
    ey2 = torch.max(py2, ty2)
    c2 = (ex2 - ex1).pow(2) + (ey2 - ey1).pow(2) + eps

    p_cx, p_cy = (px1 + px2) / 2, (py1 + py2) / 2
    t_cx, t_cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    rho2 = (p_cx - t_cx).pow(2) + (p_cy - t_cy).pow(2)

    p_w = (px2 - px1).clamp(0)
    p_h = (py2 - py1).clamp(0)
    t_w = (tx2 - tx1).clamp(0)
    t_h = (ty2 - ty1).clamp(0)

    v = (4 / (math.pi ** 2)) * (torch.atan(t_w / t_h) - torch.atan(p_w / p_h), 2).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    
    ciou = iou - (rho2 / c2) - alpha * v
    return (1 - ciou).sum()

class DetectionLoss(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES, class_counts=None, lambda_coord=5,
                 lambda_noobj=0.1, focal_gamma=2.0, focal_alpha=0.25):
        super().__init__()

        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        if class_counts is None:
            counts = torch.ones(num_classes, dtype=torch.float32)
        else:
            counts = torch.as_tensor(class_counts, dtype=torch.float32)

            if counts.numel() != num_classes or torch.any(counts <= 0):
                raise ValueError("class_counts must contain one positive count per class")

        weights = counts.sum() / (len(counts) * counts)
        weights = weights.clamp(max=50.0)
        weights = weights / weights.mean()
        self.register_buffer("class_weights", weights)

    def _decode_boxes_from_mask(self, tensor, mask, grid_size):
        row_idx, col_idx = mask.nonzero(as_tuple=True)[1], mask.nonzero(as_tuple=True)[2]
        values = tensor[mask]

        tx, ty = values[:, 1], values[:, 2]
        sqrt_w, sqrt_h = values[:, 3], values[:, 4]

        cx = (col_idx.float() + tx) / grid_size
        cy = (row_idx.float() + ty) / grid_size
        w = sqrt_w ** 2
        h = sqrt_h ** 2

        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + 2 / 2, cy + h / 2

        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def _single_scale_loss(self, pred, target):
        obj_mask = target[..., 0] == 1
        noobj_mask = target[..., 0] == 0

        if obj_mask.any():
            grid_size = pred.size(1)
            pred_xyxy = self._decode_boxes_from_mask(pred, obj_mask, grid_size)
            target_xyxy = self._decode_boxes_from_mask(target, obj_mask, grid_size)
            coord_loss = ciou_loss(pred_xyxy, target_xyxy)
        else:
            coord_loss = pred.new_zeros(())

        def focal_bce_with_logits(logits, targets, alpha, gamma):
            if logits.numel() == 0:
                return logits.new_zeros(())
            
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            probs = torch.sigmoid()
            pt = probs * targets + (1.0 - probs) * (1.0 - targets)
            alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
            focal_weight = alpha_factor * (1.0 - pt).pow(gamma)

            return (focal_weight * bce).sum()
        
        obj_logits = pred[obj_mask][..., 0]
        noobj_logits = pred[noobj_mask][..., 0]

        obj_targets = target[obj_mask][..., 0]
        noobj_targets = target[noobj_mask][..., 0]

        obj_loss = focal_bce_with_logits(obj_logits, obj_targets, self.focal_alpha, self.focal_gamma)
        noobj_loss = focal_bce_with_logits(noobj_logits, noobj_targets, self.focal_alpha, self.focal_gamma)

        pred_cls = pred[obj_mask][..., 5:]
        target_cls = target[obj_mask][..., 5:]
        if pred_cls.numel() == 0:
            class_loss = pred.new_zeros(())
        else:
            target_cls_ids = target_cls.argmax(dim=-1)
            class_loss = F.cross_entropy(
                pred_cls,
                target_cls,
                weight=self.class_weights,
                reduction='sum'
            )

        batch_size = pred.size(0)
        total = (self.lambda_coord * coord_loss + obj_loss + self.lambda_noobj * noobj_loss + class_loss) / batch_size

        return total, {
            "coord": coord_loss.item() / batch_size,
            "obj": obj_loss.item() / batch_size,
            "noobj_loss": noobj_loss.item() / batch_size,
            "class": class_loss.item() / batch_size
        }

    def forward(self, pred, targets):

        if not isinstance(pred, dict) or not isinstance(targets, dict):
            raise TypeError("DetectionLoss expects dictionaries of fine and coarse predictions/targets")
        
        total = None
        stats = {"coord": 0.0, "obj": 0.0, "noobj": 0.0, "class": 0.0}
        for scale in ("fine", "coarse"):
            scale_loss, scale_stats = self._single_scale_loss(pred[scale], targets[scale])
            total = scale_loss if total is None else total + scale_loss
            for key in stats:
                stats[key] += scale_stats[key]

        return total, stats