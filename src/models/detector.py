import torch.nn as nn
import torch

from src.config.configs import BOXES_PER_CELL, NUM_CLASSES

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

class Detector(nn.Module):
    
    def __init__(self, num_classes=NUM_CLASSES, boxes_per_cell=BOXES_PER_CELL):
        super().__init__()
        self.num_classes = num_classes
        self.boxes_per_cell = boxes_per_cell

        self.backbone = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512)
        )

        self.dropout = nn.Dropout2d(0.2)
        self.head = nn.Conv2d(512, boxes_per_cell * (5 + num_classes), kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feat = self.backbone(x)
        pred = self.head(self.dropout(feat))
        batch_size, _, grid_h, grid_w = pred.shape
        pred = pred.view(batch_size, self.boxes_per_cell, 5 + self.num_classes, grid_h, grid_w)
        pred = pred.permute(0, 3, 4, 1, 2)

        obj = pred[..., 0:1].sigmoid()
        txty = pred[..., 1:3].sigmoid()
        twth = pred[..., 3:5]
        class_probs = pred[..., 5:].softmax(dim=-1)

        return torch.cat([obj, txty, twth, class_probs], dim=-1)
