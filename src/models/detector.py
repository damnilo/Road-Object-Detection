import torch.nn as nn
import torch

from src.config.configs import NUM_CLASSES

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

class Detector(nn.Module):
    
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.backbone = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512)
        )

        self.head = nn.Conv2d(512, (5 + num_classes), kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feat = self.backbone(x)
        pred = self.head(feat)
        pred = pred.permute(0, 2, 3, 1)

        obj = pred[..., 0:1].sigmoid()
        txty = pred[..., 1:3].sigmoid()
        twth = pred[..., 3:5]
        class_probs = pred[..., 5:].softmax(dim=-1)

        return torch.cat([obj, txty, twth, class_probs], dim=-1)