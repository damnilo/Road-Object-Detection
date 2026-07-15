import cv2
import torch

from src.detection.bbox import agnostic_nms, decode_multiscale_predictions, iou, nms
from src.config.configs import IMG_SIZE, FINE_GRID_SIZE, GRID_SIZE, NUM_CLASSES, CLASSES
from src.data.transform import letterbox_image
from src.models.detector import Detector


class CarDetection:
    
    def __init__(self, weights_path=None, device='cpu', conf_thresh=0.5, iou_thresh=0.45):
        self.device = device
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        self.model = Detector(num_classes=NUM_CLASSES).to(self.device)
        if weights_path:
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def detect(self, frame):

        orig_h, orig_w, _ = frame.shape

        img, letterbox_transform = letterbox_image(frame, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        predictions = self.model(tensor)

        boxes_xyxy, scores, labels = decode_multiscale_predictions(
            {scale: prediction[0] for scale, prediction in predictions.items()},
            {'fine': FINE_GRID_SIZE, 'coarse': GRID_SIZE}, NUM_CLASSES,
            conf_threshold=self.conf_thresh,
        )

        if boxes_xyxy.size(0) == 0:
            return []
        
        keep = nms(boxes_xyxy, scores, labels, iou_threshold=self.iou_thresh)
        boxes_xyxy, scores, labels = boxes_xyxy[keep], scores[keep], labels[keep]

        keep2 = agnostic_nms(boxes_xyxy, scores, iou_threshold=self.iou_thresh)
        boxes_xyxy, scores, labels = boxes_xyxy[keep2], scores[keep2], labels[keep2]

        results = []
        scale_x, scale_y, pad_left, pad_top = letterbox_transform
        for box, score, label in zip(boxes_xyxy, scores, labels):
            x1, y1, x2, y2 = box.tolist()

            x1 = int(max(0, min(orig_w, (x1 * IMG_SIZE - pad_left) / scale_x)))
            y1 = int(max(0, min(orig_h, (y1 * IMG_SIZE - pad_top) / scale_y)))
            x2 = int(max(0, min(orig_w, (x2 * IMG_SIZE - pad_left) / scale_x)))
            y2 = int(max(0, min(orig_h, (y2 * IMG_SIZE - pad_top) / scale_y)))

            results.append({
                'box': (x1, y1, x2, y2),
                'score': score.item(),
                'label': CLASSES[label.item()]
            })

        return results
