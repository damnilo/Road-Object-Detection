import torch.nn as nn
import torch

from src.config.configs import BOXES_PER_CELL, NUM_CLASSES

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )

class Detector(nn.Module):
    
    def __init__(self, num_classes=NUM_CLASSES, boxes_per_cell=BOXES_PER_CELL):
        super().__init__()
        self.num_classes = num_classes
        self.boxes_per_cell = boxes_per_cell

        self.block1 = conv_block(3, 32)
        self.block2 = conv_block(32, 64)
        self.block3 = conv_block(64, 128)
        self.block4 = conv_block(128, 256)
        self.block5 = conv_block(256, 512)

        output_channels = boxes_per_cell * (5 + num_classes)
        self.fine_lateral = nn.Conv2d(512, 256, kernel_size=1)
        self.fine_refine = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fine_head = nn.Conv2d(256, output_channels, kernel_size=1)
        self.coarse_head = nn.Conv2d(512, output_channels, kernel_size=1)

    def forward(self, x):
        feat = self.block1(x)
        feat = self.block2(feat)
        feat = self.block3(feat)
        fine_feat = self.block4(feat)
        coarse_feat = self.block5(fine_feat)

        fine_context = self.fine_lateral(coarse_feat)
        fine_context = torch.nn.functional.interpolate(
            fine_context,
            size=fine_feat.shape[-2:],
            mode='nearest',
        )
        fine_feat = self.fine_refine(fine_feat + fine_context)

        return {
            "fine": self._format_predictions(self.fine_head(fine_feat)),
            "coarse": self._format_predictions(self.coarse_head(coarse_feat)),
        }

    def _format_predictions(self, pred):
        batch_size, _, grid_h, grid_w = pred.shape
        pred = pred.view(batch_size, self.boxes_per_cell, 5 + self.num_classes, grid_h, grid_w)
        pred = pred.permute(0, 3, 4, 1, 2)

        obj = pred[..., 0:1]
        txty = pred[..., 1:3].sigmoid()
        twth = pred[..., 3:5]
        class_logits = pred[..., 5:]

        return torch.cat([obj, txty, twth, class_logits], dim=-1)
