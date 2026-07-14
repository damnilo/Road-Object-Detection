import cv2

def draw_boxes(frame, boxes):

    for box in boxes:

        x1 = int(box.x)
        y1 = int(box.y)

        x2 = int(box.x + box.width)
        y2 = int(box.y + box.height)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"{box.class_name}: {box.confidence:.2f}"

        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame