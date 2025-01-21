import cv2
import numpy as np
import random
import json
import time

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
        
        # Larger button dimensions
        self.button_submit = {'x': 50, 'y': 30, 'w': 150, 'h': 50}
        self.button_skip = {'x': 250, 'y': 30, 'w': 150, 'h': 50}
        
        cv2.namedWindow('Ball Calibration')
        cv2.setMouseCallback('Ball Calibration', self.mouse_callback)

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
        # Draw Submit button
        cv2.rectangle(frame, 
                     (self.button_submit['x'], self.button_submit['y']), 
                     (self.button_submit['x'] + self.button_submit['w'], 
                      self.button_submit['y'] + self.button_submit['h']), 
                     (0, 255, 0), -1)
        cv2.putText(frame, 'Submit', 
                   (self.button_submit['x'] + 20, self.button_submit['y'] + 25),
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
            cv2.circle(frame, self.current_point, 5, (0, 255, 0), -1)
            cv2.circle(frame, self.current_point, 20, (0, 255, 0), 2)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is on skip button first
            if (self.button_skip['x'] <= x <= self.button_skip['x'] + self.button_skip['w'] and
                  self.button_skip['y'] <= y <= self.button_skip['y'] + self.button_skip['h']):
                self.calibration_count += 1
                self.current_point = None
                print(f"Frame {self.calibration_count} skipped")
                # Force break from the current frame loop
                self.force_next_frame = True
                return

            # Check submit button
            elif (self.button_submit['x'] <= x <= self.button_submit['x'] + self.button_submit['w'] and
                self.button_submit['y'] <= y <= self.button_submit['y'] + self.button_submit['h']):
                if self.current_point:  # Only submit if we have a point
                    self.calibration_points.append({
                        'frame_number': self.current_frame_number,
                        'point': self.current_point,
                        'timestamp': time.time()
                    })
                    self.calibration_count += 1
                    self.current_point = None
                    print(f"Calibration {self.calibration_count} saved")
                    self.save_results()
                    # Force break from the current frame loop
                    self.force_next_frame = True
                    return
            else:
                # Set new point
                self.current_point = (x, y)
                print(f"Ball position marked at {x}, {y}")

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
                'session_id': time.strftime("%Y%m%d_%H%M%S")  # Add session identifier
            }
            existing_results.append(new_point)
        
        # Save combined results
        with open('ball_calibration.json', 'w') as f:
            json.dump(existing_results, f, indent=4)
        print(f"Results appended to ball_calibration.json (Total entries: {len(existing_results)})")

    def calibrate(self):
        self.force_next_frame = False
        
        while self.calibration_count < self.max_calibrations:
            # Get random frame for each calibration
            frame, frame_number = self.get_random_frame()
            if frame is None:
                print("Error: Could not get valid frame, trying to reset video capture")
                self.video.release()
                self.video = cv2.VideoCapture("barrea.mp4")
                if not self.video.isOpened():
                    print("Fatal error: Could not reopen video")
                    break
                continue
            
            try:
                self.current_frame = frame.copy()
                self.current_frame_number = frame_number
                self.force_next_frame = False
                
                while not self.force_next_frame:
                    display_frame = self.current_frame.copy()
                    
                    # Draw current point
                    self.draw_point(display_frame)
                    
                    # Draw buttons
                    self.draw_buttons(display_frame)
                    
                    # Show calibration progress and instructions
                    cv2.putText(display_frame, 
                               f'Calibration: {self.calibration_count + 1}/{self.max_calibrations}',
                               (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.putText(display_frame, 
                               'Click on ball or press Skip if no ball visible',
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
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
        print("2. Click 'Submit' to save the position")
        print("3. Click 'Skip' if ball is not visible")
        print("4. Repeat for 5 different frames")
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