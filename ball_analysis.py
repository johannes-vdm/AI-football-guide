import cv2
import numpy as np
import random
import json
import time

class BallTracker:
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.calibration_points = []
        self.current_frame = None
        self.current_point = None
        self.calibration_count = 0
        self.max_calibrations = 5
        
        # Button dimensions
        self.button_submit = {'x': 50, 'y': 30, 'w': 100, 'h': 40}
        self.button_skip = {'x': 170, 'y': 30, 'w': 100, 'h': 40}
        
        cv2.namedWindow('Ball Calibration')
        cv2.setMouseCallback('Ball Calibration', self.mouse_callback)

    def get_random_frame(self):
        random_frame = random.randint(0, self.total_frames - 1)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        ret, frame = self.video.read()
        if ret:
            return frame, random_frame
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
            # Check if click is on buttons
            if (self.button_submit['x'] <= x <= self.button_submit['x'] + self.button_submit['w'] and
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
                    self.save_results()  # Save after each submission
                    return
            elif (self.button_skip['x'] <= x <= self.button_skip['x'] + self.button_skip['w'] and
                  self.button_skip['y'] <= y <= self.button_skip['y'] + self.button_skip['h']):
                # Skip this frame without requiring a point
                self.calibration_count += 1
                self.current_point = None
                print(f"Frame {self.calibration_count} skipped")
                return
            else:
                # Set new point
                self.current_point = (x, y)
                print(f"Ball position marked at {x}, {y}")

    def save_results(self):
        results = []
        for point in self.calibration_points:
            results.append({
                'frame_number': point['frame_number'],
                'x': point['point'][0],
                'y': point['point'][1],
                'timestamp': point['timestamp']
            })
        
        with open('ball_calibration.json', 'w') as f:
            json.dump(results, f, indent=4)
        print("Results saved to ball_calibration.json")

    def calibrate(self):
        while self.calibration_count < self.max_calibrations:
            # Get random frame for each calibration
            frame, frame_number = self.get_random_frame()
            if frame is None:
                break
            
            self.current_frame = frame.copy()
            self.current_frame_number = frame_number
            
            while True:  # Loop for current frame until submitted or skipped
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
                           'Click on the ball',
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Ball Calibration', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.save_results()
                    return self.calibration_points
                
                # Break inner loop if calibration was submitted or skipped
                if len(self.calibration_points) > self.calibration_count or \
                   self.calibration_count > len(self.calibration_points):
                    break
                
                time.sleep(0.03)
        
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