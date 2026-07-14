import cv2

class VideoLoader:

    def __init__(self, video_path):

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
    def read(self):

        ret, frame = self.cap.read()

        if not ret:
            return None
        
        return frame
    
    def release(self):
        
        self.cap.release()