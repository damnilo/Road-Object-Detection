import cv2

def draw_boxes(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = box['box']

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"{box['label']}: {box['score']:.2f}"

        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame
