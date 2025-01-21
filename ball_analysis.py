import cv2
import numpy as np
import random
import json
import time
import torch
from torchvision.models import detection
from torchvision.transforms import functional as F
from PIL import Image

class BallTracker:
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            raise Exception("Error: Could not open video file")
        
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {self.total_frames}")
        
        # Start from beginning of video
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self.calibration_points = []
        self.current_frame = None
        self.current_point = None
        self.calibration_count = 0
        self.max_calibrations = 5
        self.force_next_frame = False
        
        # Redesign button layout
        self.button_ai_correct = {'x': 50, 'y': 30, 'w': 150, 'h': 50}
        self.button_ai_wrong = {'x': 220, 'y': 30, 'w': 150, 'h': 50}
        self.button_skip = {'x': 390, 'y': 30, 'w': 150, 'h': 50}
        
        cv2.namedWindow('Ball Calibration')
        cv2.setMouseCallback('Ball Calibration', self.mouse_callback)
        
        self.setup_model()

    def setup_model(self):
        print("Loading PyTorch model...")
        # Load pre-trained Faster R-CNN with updated parameter
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')  # Changed from pretrained=True
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def get_random_frame(self):
        max_attempts = 10
        for _ in range(max_attempts):
            # Get a random frame from first 80% of the video to avoid end-of-file issues
            random_frame = random.randint(0, int(self.total_frames * 0.8))
            
            # Seek to frame
            self.video.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
            
            # Try to read frame
            ret, frame = self.video.read()
            if ret and frame is not None and frame.size > 0:
                print(f"Successfully loaded frame {random_frame}")
                return frame, random_frame
            
            print(f"Failed to load frame {random_frame}, retrying...")
        
        print("Failed to get valid random frame after multiple attempts")
        # If all attempts fail, try reading the next available frame
        ret, frame = self.video.read()
        if ret and frame is not None and frame.size > 0:
            current_frame = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
            return frame, current_frame
        
        return None, None

    def draw_buttons(self, frame):
        # Draw AI Correct button
        cv2.rectangle(frame, 
                     (self.button_ai_correct['x'], self.button_ai_correct['y']), 
                     (self.button_ai_correct['x'] + self.button_ai_correct['w'], 
                      self.button_ai_correct['y'] + self.button_ai_correct['h']), 
                     (0, 255, 0), -1)
        cv2.putText(frame, 'AI Correct', 
                   (self.button_ai_correct['x'] + 20, self.button_ai_correct['y'] + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw AI Wrong button
        cv2.rectangle(frame, 
                     (self.button_ai_wrong['x'], self.button_ai_wrong['y']),
                     (self.button_ai_wrong['x'] + self.button_ai_wrong['w'], 
                      self.button_ai_wrong['y'] + self.button_ai_wrong['h']),
                     (0, 255, 255), -1)
        cv2.putText(frame, 'AI Wrong',
                   (self.button_ai_wrong['x'] + 30, self.button_ai_wrong['y'] + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw Skip button
        cv2.rectangle(frame, 
                     (self.button_skip['x'], self.button_skip['y']),
                     (self.button_skip['x'] + self.button_skip['w'], 
                      self.button_skip['y'] + self.button_skip['h']),
                     (0, 0, 255), -1)
        cv2.putText(frame, 'Skip',
                   (self.button_skip['x'] + 30, self.button_skip['y'] + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    def draw_point(self, frame):
        if self.current_point:
            cv2.circle(frame, self.current_point, 2, (0, 255, 0), -1)  # Smaller center dot
            cv2.circle(frame, self.current_point, 10, (0, 255, 0), 1)  # Smaller outer circle

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"\nClick detected at x:{x}, y:{y}")
            
            # Check AI Correct button
            if (self.button_ai_correct['x'] <= x <= self.button_ai_correct['x'] + self.button_ai_correct['w'] and
                self.button_ai_correct['y'] <= y <= self.button_ai_correct['y'] + self.button_ai_correct['h']):
                print("AI Correct button clicked!")
                ball_center, confidence = self.detect_ball(self.current_frame)
                if ball_center:
                    self.calibration_points.append({
                        'frame_number': self.current_frame_number,
                        'x': ball_center[0],
                        'y': ball_center[1],
                        'timestamp': time.time(),
                        'ai_confidence': confidence,
                        'ai_validated': True
                    })
                    self.calibration_count += 1
                    print(f"AI prediction saved with confidence {confidence:.2f}")
                    self.save_results()
                    self.force_next_frame = True
                return

            # Check AI Wrong button
            elif (self.button_ai_wrong['x'] <= x <= self.button_ai_wrong['x'] + self.button_ai_wrong['w'] and
                  self.button_ai_wrong['y'] <= y <= self.button_ai_wrong['y'] + self.button_ai_wrong['h']):
                print("AI Wrong button clicked!")
                self.current_point = None  # Reset point for manual selection
                print("Please click on the correct ball position")
                return

            # Check Skip button
            elif (self.button_skip['x'] <= x <= self.button_skip['x'] + self.button_skip['w'] and
                  self.button_skip['y'] <= y <= self.button_skip['y'] + self.button_skip['h']):
                print("Skip button clicked!")
                self.calibration_count += 1
                self.current_point = None
                self.force_next_frame = True
                print(f"Frame {self.calibration_count} skipped")
                return

            # Handle manual ball position click
            else:
                self.current_point = (x, y)
                print(f"Manual ball position marked at {x}, {y}")
                # Save manual position
                self.calibration_points.append({
                    'frame_number': self.current_frame_number,
                    'x': x,
                    'y': y,
                    'timestamp': time.time(),
                    'ai_validated': False
                })
                self.calibration_count += 1
                self.save_results()
                self.force_next_frame = True

    def save_results(self):
        # Load existing results if file exists
        existing_results = []
        try:
            with open('ball_calibration.json', 'r') as f:
                existing_results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_results = []
        
        # Add new results
        for point in self.calibration_points:
            new_point = {
                'frame_number': point['frame_number'],
                'x': point['point'][0],
                'y': point['point'][1],
                'timestamp': point['timestamp'],
                'session_id': time.strftime("%Y%m%d_%H%M%S"),  # Add session identifier
                'ai_confidence': point['ai_confidence'],
                'ai_validated': point['ai_validated']
            }
            existing_results.append(new_point)
        
        # Save combined results
        with open('ball_calibration.json', 'w') as f:
            json.dump(existing_results, f, indent=4)
        print(f"Results appended to ball_calibration.json (Total entries: {len(existing_results)})")

    def detect_ball(self, frame):
        # Convert frame to PyTorch tensor
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Get predictions for sports ball class (index 37 in COCO dataset)
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter for ball detections (COCO class 37) with high confidence
        ball_detections = [(box, score) for box, score, label in 
                          zip(boxes, scores, labels) if label == 37 and score > 0.7]
        
        if ball_detections:
            # Get the highest confidence detection
            box, score = max(ball_detections, key=lambda x: x[1])
            # Calculate center point
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            return (center_x, center_y), score
        
        return None, 0.0

    def draw_frame(self, frame):
        display_frame = frame.copy()
        
        # Try automatic ball detection
        ball_center, confidence = self.detect_ball(frame)
        
        if ball_center:
            # Draw suggested point in blue (smaller)
            cv2.circle(display_frame, ball_center, 2, (255, 0, 0), -1)  # Smaller center dot
            cv2.circle(display_frame, ball_center, 10, (255, 0, 0), 1)  # Smaller outer circle
            cv2.putText(display_frame, 
                       f'AI Confidence: {confidence:.2f}',
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw user's point in green if it exists
        if self.current_point:
            cv2.circle(display_frame, self.current_point, 2, (0, 255, 0), -1)  # Smaller center dot
            cv2.circle(display_frame, self.current_point, 10, (0, 255, 0), 1)  # Smaller outer circle
        
        # Draw buttons
        self.draw_buttons(display_frame)
        
        # Show calibration progress and instructions
        cv2.putText(display_frame, 
                   f'Calibration: {self.calibration_count + 1}/{self.max_calibrations}',
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(display_frame, 
                   'Click to correct AI detection or Skip if no ball visible',
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display_frame

    def calibrate(self):
        self.force_next_frame = False
        
        while self.calibration_count < self.max_calibrations:
            frame, frame_number = self.get_random_frame()
            if frame is None:
                print("Error: Could not get valid frame")
                continue
            
            try:
                self.current_frame = frame.copy()
                self.current_frame_number = frame_number
                self.force_next_frame = False
                
                # Auto-detect ball and set as current point
                ball_center, confidence = self.detect_ball(frame)
                if ball_center and confidence > 0.7:
                    self.current_point = ball_center
                
                while not self.force_next_frame:
                    display_frame = self.draw_frame(self.current_frame)
                    cv2.imshow('Ball Calibration', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.save_results()
                        return self.calibration_points
                    
                    time.sleep(0.03)
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
        
        cv2.destroyAllWindows()
        self.save_results()
        return self.calibration_points

    def start(self):
        print("Starting calibration mode...")
        print("Instructions:")
        print("1. Click on the ball in the frame")
        print("2. Click 'AI Correct' to save the position")
        print("3. Click 'AI Wrong' if ball is not visible")
        print("4. Click 'Skip' if ball is not visible")
        print("5. Repeat for 5 different frames")
        print("Press 'q' to quit at any time")
        print("Results will be saved to ball_calibration.json")
        
        calibration_data = self.calibrate()
        
        if calibration_data:
            print("\nCalibration completed!")
            print(f"Collected {len(calibration_data)} ball positions")
        
        self.video.release()

if __name__ == "__main__":
    tracker = BallTracker("barrea.mp4")
    tracker.start() 