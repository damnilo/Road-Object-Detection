import torch
from dataclasses import dataclass

@dataclass
class BoundingBox:

    x: float
    y: float

    width: float
    height: float

    confidence: float

    class_name: str

    def center(self):
        return (self.x + self.width / 2, self.y + self.height / 2)
    