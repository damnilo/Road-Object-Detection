import cv2
import argparse
import os

from src.data.video_loader import VideoLoader
from src.visualization.draw import draw_boxes
from src.models.car_detection import CarDetection

def main(video_path, weights_path, device='cpu'):
    video = VideoLoader(video_path)

    detector = CarDetection(weights_path=weights_path, device=device, conf_thresh=0.40, iou_thresh=0.45)

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
    parser = argparse.ArgumentParser(description='Run car detection on a video.')
    parser.add_argument('video_path', nargs='?', help='Path to the input video file.')
    parser.add_argument('--video-path', dest='video_path_flag', help='Path to the input video file.')
    parser.add_argument('--weights', default='checkpoints/best_detector_weights.pth')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    video_path = args.video_path_flag or args.video_path
    if not video_path:
        parser.error("the following argument is required: video_path (or --video-path)")

    if not os.path.exists(video_path):
        parser.error(f"video file not found: {video_path}")

    main(video_path, args.weights, args.device)
