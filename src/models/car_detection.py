import cv2
import torch

from src.detection.bounding_box import BoundingBox
from src.config.configs import IMG_SIZE, GRID_SIZE, NUM_CLASSES, CLASSES
from src.models.detector import Detector


class CarDetection:
    
    def __init__(self, weights_path=None, device='cpu', conf_thresh=0.5, iou_thresh=0.45):
        self.device = device
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.bbox = BoundingBox()

        self.model = Detector(num_classes=NUM_CLASSES).to(self.device)
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def detect(self, frame):

        orig_h, orig_w, _ = frame.shape

        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        pred = self.model(tensor)[0]

        boxes_xyxy, scores, labels = self.bbox.decode_predictions(
            pred, GRID_SIZE, NUM_CLASSES, conf_threshold=self.conf_thresh
        )

        if boxes_xyxy.size(0) == 0:
            return []
        
        keep = self.bbox.nms(boxes_xyxy, scores, labels, iou_threshold=self.iou_thresh)

        results = []
        for idx in keep:
            box = boxes_xyxy[idx]
            score = scores[idx].item()
            label = CLASSES[labels[idx].item()]

            x1, y1, x2, y2 = box
            x1 = int(x1 * orig_w)
            y1 = int(y1 * orig_h)
            x2 = int(x2 * orig_w)
            y2 = int(y2 * orig_h)

            results.append({
                'box': (x1, y1, x2, y2),
                'score': score,
                'label': label
            })

        return results