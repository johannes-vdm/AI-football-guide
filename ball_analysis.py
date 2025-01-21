import cv2
import numpy as np
from scipy.spatial import distance
import time
from collections import deque
import imutils
import threading
from queue import Queue

class BallTracker:
    def __init__(self, video_path, ball_diameter_cm=22, buffer=64, start_minute=10):
        self.video = cv2.VideoCapture(video_path)
        self.calibration_points = []
        self.current_frame = None
        self.dragging = False
        self.drag_start = None
        self.drag_current = None
        self.calibration_mode = True
        self.calibration_count = 0
        self.max_calibrations = 5
        
        # Button dimensions
        self.button_yes = {'x': 50, 'y': 30, 'w': 100, 'h': 40}
        self.button_no = {'x': 170, 'y': 30, 'w': 100, 'h': 40}
        
        # Set video position to 10 minutes
        fps = self.video.get(cv2.CAP_PROP_FPS)
        frame_count = int(fps * start_minute * 60)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        
        cv2.namedWindow('Ball Calibration')
        cv2.setMouseCallback('Ball Calibration', self.mouse_callback)
        
        self.ball_diameter_cm = ball_diameter_cm
        self.pixels_per_cm = 1.0
        self.pts = deque(maxlen=buffer)
        self.frame_queue = Queue(maxsize=128)
        self.results_queue = Queue()
        self.running = True
        
        # Adjusted HSV range for white football in match conditions
        self.lower_ball = np.array([0, 0, 200])  # Higher value threshold for white
        self.upper_ball = np.array([180, 50, 255])  # Increased saturation range
        
        # Add minimum and maximum ball size (in pixels)
        self.min_ball_radius = 3
        self.max_ball_radius = 30
        
        self.process_thread = threading.Thread(target=self._process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()

    def _process_frames(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                ball_pos, center = self.detect_ball(frame)
                self.results_queue.put((ball_pos, center))
            else:
                time.sleep(0.001)  # Short sleep to prevent CPU overload

    def detect_ball(self, frame):
        # Preserve original frame for display
        debug_frame = frame.copy()
        
        # Resize for consistent processing
        frame = imutils.resize(frame, width=1280)  # Increased resolution
        
        # Enhanced preprocessing
        blurred = cv2.GaussianBlur(frame, (9, 9), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Create and clean up mask
        mask = cv2.inRange(hsv, self.lower_ball, self.upper_ball)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Show mask for debugging
        cv2.imshow('Mask', mask)
        
        # Find contours
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        # Sort contours by area, largest first
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        
        for c in cnts[:5]:  # Check top 5 largest contours
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            
            # Filter by size
            if self.min_ball_radius < radius < self.max_ball_radius:
                # Check circularity
                area = cv2.contourArea(c)
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > 0.7:  # Circle check
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                        self.pixels_per_cm = (radius * 2) / self.ball_diameter_cm
                        return (int(x - radius), int(y - radius), 
                               int(radius * 2), int(radius * 2)), center
        
        return None, None

    def calculate_speed(self, pos1, pos2, time_diff):
        if pos1 is None or pos2 is None or time_diff == 0:
            return 0
        
        center1 = (pos1[0] + pos1[2]/2, pos1[1] + pos1[3]/2)
        center2 = (pos2[0] + pos2[2]/2, pos2[1] + pos2[3]/2)
        
        dist_pixels = distance.euclidean(center1, center2)
        dist_meters = (dist_pixels / self.pixels_per_cm) / 100
        
        speed = dist_meters / time_diff
        # Add reasonable speed limit (e.g., max 200 km/h = ~55 m/s)
        return min(speed, 55.0)

    def analyze_video(self):
        prev_pos = None
        prev_time = None
        fps = self.video.get(cv2.CAP_PROP_FPS)
        frame_time = 1/fps
        
        while True:
            start_time = time.time()
            
            ret, frame = self.video.read()
            if not ret:
                break
            
            if not self.frame_queue.full():
                self.frame_queue.put(frame.copy())
            
            if not self.results_queue.empty():
                ball_pos, center = self.results_queue.get()
                
                if ball_pos:
                    x, y, w, h = ball_pos
                    # Draw both rectangle and circle for better visualization
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                    cv2.circle(frame, (int(x + w/2), int(y + h/2)), int(w/2),
                              (0, 255, 255), 1)
                    cv2.circle(frame, center, 3, (0, 0, 255), -1)
                    
                    if prev_pos and prev_time:
                        speed = self.calculate_speed(prev_pos, ball_pos, 
                                                  time.time() - prev_time)
                        cv2.putText(frame, f'Speed: {speed:.2f} m/s', 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    self.pts.appendleft(center)
                    prev_pos = ball_pos
                    prev_time = time.time()
                    
                    # Draw trail
                    for i in range(1, len(self.pts)):
                        if self.pts[i - 1] is None or self.pts[i] is None:
                            continue
                        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                        cv2.line(frame, self.pts[i - 1], self.pts[i], 
                                (0, 0, 255), thickness)
            
            cv2.imshow('Ball Tracking', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):  # Add pause functionality
                cv2.waitKey(0)
            
            elapsed_time = time.time() - start_time
            if elapsed_time < frame_time:
                time.sleep(frame_time - elapsed_time)
        
        self.running = False
        self.video.release()
        cv2.destroyAllWindows()

    def draw_buttons(self, frame):
        # Draw "Ball Found" button
        cv2.rectangle(frame, 
                     (self.button_yes['x'], self.button_yes['y']), 
                     (self.button_yes['x'] + self.button_yes['w'], 
                      self.button_yes['y'] + self.button_yes['h']), 
                     (0, 255, 0), -1)
        cv2.putText(frame, 'Ball Found', 
                   (self.button_yes['x'] + 10, self.button_yes['y'] + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw "No Ball" button
        cv2.rectangle(frame, 
                     (self.button_no['x'], self.button_no['y']),
                     (self.button_no['x'] + self.button_no['w'], 
                      self.button_no['y'] + self.button_no['h']),
                     (0, 0, 255), -1)
        cv2.putText(frame, 'No Ball',
                   (self.button_no['x'] + 20, self.button_no['y'] + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    def mouse_callback(self, event, x, y, flags, param):
        if self.calibration_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if click is on buttons
                if (self.button_yes['x'] < x < self.button_yes['x'] + self.button_yes['w'] and
                    self.button_yes['y'] < y < self.button_yes['y'] + self.button_yes['h']):
                    if self.drag_current and self.drag_start:
                        self.calibration_points.append({
                            'start': self.drag_start,
                            'end': self.drag_current,
                            'frame': self.current_frame.copy()
                        })
                        self.calibration_count += 1
                        print(f"Calibration {self.calibration_count} saved")
                elif (self.button_no['x'] < x < self.button_no['x'] + self.button_no['w'] and
                      self.button_no['y'] < y < self.button_no['y'] + self.button_no['h']):
                    self.calibration_count += 1
                    print(f"Frame {self.calibration_count} marked as no ball")
                else:
                    self.dragging = True
                    self.drag_start = (x, y)
                    self.drag_current = (x, y)
            
            elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
                self.drag_current = (x, y)
            
            elif event == cv2.EVENT_LBUTTONUP:
                self.dragging = False

    def calibrate(self):
        while self.calibration_count < self.max_calibrations:
            ret, frame = self.video.read()
            if not ret:
                break
            
            self.current_frame = frame.copy()
            display_frame = frame.copy()
            
            # Draw current drag rectangle
            if self.dragging and self.drag_start and self.drag_current:
                cv2.rectangle(display_frame, self.drag_start, self.drag_current, (0, 255, 0), 2)
            
            # Draw buttons
            self.draw_buttons(display_frame)
            
            # Show calibration progress
            cv2.putText(display_frame, f'Calibration: {self.calibration_count}/{self.max_calibrations}',
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Ball Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # Add small delay to prevent high CPU usage
            time.sleep(0.03)
        
        cv2.destroyAllWindows()
        return self.calibration_points

    def start(self):
        print("Starting calibration mode...")
        print("Instructions:")
        print("1. Draw rectangle around the ball by clicking and dragging")
        print("2. Click 'Ball Found' to confirm the selection")
        print("3. Click 'No Ball' if ball is not visible in frame")
        print("4. Repeat for 5 different frames")
        calibration_data = self.calibrate()
        
        if calibration_data:
            print("\nCalibration completed!")
            print(f"Collected {len(calibration_data)} ball positions")
            # Here you can add code to use the calibration data
            # to improve ball detection parameters
        
        self.video.release()

if __name__ == "__main__":
    tracker = BallTracker("barrea.mp4", start_minute=10)
    tracker.start() 