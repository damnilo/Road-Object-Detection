import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.configs import NUM_CLASSES

class DetectionLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, class_counts=None, lambda_coord=5, lambda_noobj=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.mse = nn.MSELoss(reduction='sum')
        if class_counts is None:
            counts = torch.ones(num_classes, dtype=torch.float32)
        else:
            counts = torch.as_tensor(class_counts, dtype=torch.float32)
            if counts.numel() != num_classes or torch.any(counts <= 0):
                raise ValueError("class_counts must contain one positive count per class")
        weights = counts.sum() / (len(counts) * counts)
        weights = weights.clamp(max=5.0)
        weights = weights / weights.mean()
        self.register_buffer("class_weights", weights)

    def _single_scale_loss(self, pred, target):
        obj_mask = target[..., 0] == 1
        noobj_mask = target[..., 0] == 0

        if obj_mask.any():
            coord_loss = self.mse(pred[obj_mask][..., 1:5], target[obj_mask][..., 1:5])
        else:
            coord_loss = pred.new_zeros(())

        obj_loss = F.binary_cross_entropy(pred[obj_mask][..., 0], target[obj_mask][..., 0], reduction='sum')
        noobj_loss = F.binary_cross_entropy(pred[noobj_mask][..., 0], target[noobj_mask][..., 0], reduction='sum')

        pred_cls = pred[obj_mask][..., 5:]
        target_cls = target[obj_mask][..., 5:]
        if pred_cls.numel() == 0:
            class_loss = pred.new_zeros(())
        else:
            target_class_ids = target_cls.argmax(dim=-1)
            class_loss = F.nll_loss(
                pred_cls.clamp_min(1e-8).log(),
                target_class_ids,
                weight=self.class_weights,
                reduction='sum',
            )

        batch_size = pred.size(0)
        total = (self.lambda_coord * coord_loss + obj_loss
                 + self.lambda_noobj * noobj_loss + class_loss) / batch_size
        
        return total, {
            "coord": coord_loss.item() / batch_size,
            "obj": obj_loss.item() / batch_size,
            "noobj": noobj_loss.item() / batch_size,
            "class": class_loss.item() / batch_size
        }

    def forward(self, predictions, targets):
        if not isinstance(predictions, dict) or not isinstance(targets, dict):
            raise TypeError("DetectionLoss expects dictionaries of fine and coarse predictions/targets")

        total = None
        stats = {"coord": 0.0, "obj": 0.0, "noobj": 0.0, "class": 0.0}
        for scale in ("fine", "coarse"):
            scale_loss, scale_stats = self._single_scale_loss(predictions[scale], targets[scale])
            total = scale_loss if total is None else total + scale_loss
            for key in stats:
                stats[key] += scale_stats[key]
        return total, stats
