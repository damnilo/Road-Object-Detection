import cv2

from data.video_loader import VideoLoader
from visualization.draw import draw_boxes
from models.car_detection import CarDetection

def main():
    video = VideoLoader("path/to/video.mp4")

    detector = CarDetection()

    while True:
        frame = video.read()

        if frame is None:
            break

        boxes = detector.detect(frame)
        frame = draw_boxes(frame, boxes)

        cv2.imshow("Car Detection", frame)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()