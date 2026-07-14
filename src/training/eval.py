import torch

from src.detection.bbox import iou, xywh_to_xyxy

def compute_ap(pred_boxes, pred_scores, pred_img_ids, gt_boxes, gt_img_ids, iou_thresh=0.5):
    if len(gt_boxes) == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0
    
    if len(pred_boxes) == 0:
        return 0.0
    
    order = pred_scores.argsort(descending=True)
    pred_boxes = pred_boxes[order]
    pred_img_ids = pred_img_ids[order]

    matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
    tp = torch.zeros(len(pred_boxes), dtype=torch.float)
    fp = torch.zeros(len(pred_boxes), dtype=torch.float)

    for i in range(len(pred_boxes)):
        same_img = (gt_img_ids == pred_img_ids[i]).nonzero(as_tuple=True)[0]

        if len(same_img) == 0:
            fp[i] = 1
            continue

        ious = iou(pred_boxes[i:i+1], gt_boxes[same_img]).squeeze(0)
        best_iou, best_gt_idx = ious.max(0)
        best_gt_idx = same_img[best_gt_idx]

        if best_iou >= iou_thresh and not matched[best_gt_idx]:
            tp[i] = 1
            matched[best_gt_idx] = True
        else:
            fp[i] = 1

    tp_cum = torch.cumsum(tp, dim=0)
    fp_cum = torch.cumsum(fp, dim=0)

    recall = tp_cum / len(gt_boxes)
    precision = tp_cum / (tp_cum + fp_cum).clamp(min=1e-6)

    ap = 0.0

    for t in torch.linspace(0, 1, steps=11):
        mask = recall >= t
        p = precision[mask].max() if mask.any() else torch.tensor(0.0)
        ap += p / 11.0

    return ap.item()

@torch.no_grad()
def evaluate(model, dataloader, grid_size, num_classes, device="cpu",
             conf_thresh=0.5, nms_iou_thresh=0.45, ap_iou_thresh=0.5):
    
    from src.detection.bbox import decode_predictions, nms

    model.eval()

    pred_boxes_by_cls = {c: [] for c in range(num_classes)}
    pred_scores_by_cls = {c: [] for c in range(num_classes)}
    pred_img_ids_by_cls = {c: [] for c in range(num_classes)}

    gt_boxes_by_cls = {c: [] for c in range(num_classes)}
    gt_img_ids_by_cls = {c: [] for c in range(num_classes)}

    img_id = 0

    for images, _, raw_boxes_batch, raw_label_batch in dataloader:
        images = images.to(device)
        preds = model(images)

        for b in range(images.size(0)):
            boxes_xyxy, scores, labels = decode_predictions(
                preds[b], grid_size, num_classes, conf_threshold=conf_thresh
            )

            if boxes_xyxy.size(0) > 0:
                keep = nms(boxes_xyxy, scores, labels, iou_threshold=nms_iou_thresh)
                boxes_xyxy = boxes_xyxy[keep]
                scores = scores[keep]
                labels = labels[keep]

                for c in range(num_classes):
                    cls_mask = labels == c
                    if cls_mask.any():
                        pred_boxes_by_cls[c].append(boxes_xyxy[cls_mask])
                        pred_scores_by_cls[c].append(scores[cls_mask])
                        pred_img_ids_by_cls[c].append(torch.full((cls_mask.sum(),), img_id, dtype=torch.long))

            gt_boxes = raw_boxes_batch[b]
            gt_labels = raw_label_batch[b]

            if len(gt_boxes) > 0:
                gt_boxes_xyxy = xywh_to_xyxy(gt_boxes)
                for c in range(num_classes):
                    cls_mask = gt_labels == c
                    if cls_mask.any():
                        gt_boxes_by_cls[c].append(gt_boxes_xyxy[cls_mask])
                        gt_img_ids_by_cls[c].append(torch.full((cls_mask.sum(),), img_id, dtype=torch.long))

            img_id += 1

    aps = {}
    for c in range(num_classes):
        pb = torch.cat(pred_boxes_by_cls[c]) if pred_boxes_by_cls[c] else torch.empty((0, 4))
        ps = torch.cat(pred_scores_by_cls[c]) if pred_scores_by_cls[c] else torch.empty((0))
        pid = torch.cat(pred_img_ids_by_cls[c]) if pred_img_ids_by_cls[c] else torch.empty((0), dtype=torch.long)

        gb = torch.cat(gt_boxes_by_cls[c]) if gt_boxes_by_cls[c] else torch.empty((0, 4))
        gid = torch.cat(gt_img_ids_by_cls[c]) if gt_img_ids_by_cls[c] else torch.empty((0), dtype=torch.long)

        aps[c] = compute_ap(pb, ps, pid, gb, gid, iou_thresh=ap_iou_thresh)
    
    mean_ap = sum(aps.values()) / len(aps) if aps else 0.0
    model.train()
    return mean_ap, aps