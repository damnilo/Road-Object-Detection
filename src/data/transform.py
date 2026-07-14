import torch
import random

def hflip(image, boxes):
    image = image.flip(-1)
    
    if len(boxes) > 0:
        boxes = boxes.clone()
        boxes[:, 0] = 1.0 - boxes[:, 0]

    return image, boxes

def random_hflip(image, boxes, p=0.5):
    if random.random() < p:
        return hflip(image, boxes)
    
    return image, boxes