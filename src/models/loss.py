import torch
import torch.nn as nn

from src.config.configs import NUM_CLASSES

class DetectionLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, class_counts=None, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.mse = nn.MSELoss(reduction='sum')
        counts = torch.tensor(class_counts, dtype=torch.float32)
        weights = counts.sum() / (len(counts) * counts)
        weights = weights.clamp(max=5.0)
        weights = weights / weights.mean()
        self.register_buffer("class_weights", weights)

    def forward(self, pred, target):
        obj_mask = target[..., 0] == 1
        noobj_mask = target[..., 0] == 0

        coord_loss = self.mse(pred[obj_mask][..., 1:5], target[obj_mask][..., 1:5])

        obj_loss = nn.functional.binary_cross_entropy(pred[obj_mask][..., 0], target[obj_mask][..., 0], reduction='sum')
        noobj_loss = nn.functional.binary_cross_entropy(pred[noobj_mask][..., 0], target[noobj_mask][..., 0], reduction='sum')

        pred_cls = pred[obj_mask][..., 5:]
        target_cls = target[obj_mask][..., 5:]
        per_elem_bce = nn.functional.binary_cross_entropy(pred_cls, target_cls, reduction='none')
        weighted_bce = per_elem_bce * self.class_weights.unsqueeze(0)
        class_loss = weighted_bce.sum()

        batch_size = pred.size(0)
        total = (self.lambda_coord * coord_loss + obj_loss
                 + self.lambda_noobj * noobj_loss + class_loss) / batch_size
        
        return total, {
            "coord": coord_loss.item() / batch_size,
            "obj": obj_loss.item() / batch_size,
            "noobj": noobj_loss.item() / batch_size,
            "class": class_loss.item() / batch_size
        }