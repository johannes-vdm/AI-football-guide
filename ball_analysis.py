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
        print("\nInitializing BallTracker...")
        try:
            self.video = cv2.VideoCapture(video_path)
            if not self.video.isOpened():
                raise Exception(f"Error: Could not open video file: {video_path}")
            
            self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video.get(cv2.CAP_PROP_FPS)
            print(f"Video loaded successfully:")
            print(f"- Total frames: {self.total_frames}")
            print(f"- FPS: {self.fps}")
            print(f"- Duration: {self.total_frames/self.fps:.2f} seconds")
            
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
            self.mouse_x = 0
            self.mouse_y = 0
            
            # Add learning tracking
            self.corrections = []
            self.frame_size = (800, 600)  # Smaller frame size for faster processing

        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

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
        # Update mouse position for crosshair
        self.mouse_x = x
        self.mouse_y = y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            try:
                print(f"\nMouse click at (x:{x}, y:{y})")
                
                # Log button bounds for debugging
                for button_name, button in [
                    ('AI Correct', self.button_ai_correct),
                    ('AI Wrong', self.button_ai_wrong),
                    ('Skip', self.button_skip)
                ]:
                    x1, y1 = button['x'], button['y']
                    x2, y2 = x1 + button['w'], y1 + button['h']
                    print(f"{button_name} button bounds: ({x1},{y1}) to ({x2},{y2})")
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        print(f"Click is within {button_name} button bounds")
                
                # Check AI Correct button
                if (self.button_ai_correct['x'] <= x <= self.button_ai_correct['x'] + self.button_ai_correct['w'] and
                    self.button_ai_correct['y'] <= y <= self.button_ai_correct['y'] + self.button_ai_correct['h']):
                    print("AI Correct button clicked!")
                    if hasattr(self, 'current_ball_center') and hasattr(self, 'current_confidence'):
                        self.calibration_points.append({
                            'frame_number': self.current_frame_number,
                            'x': self.current_ball_center[0],
                            'y': self.current_ball_center[1],
                            'timestamp': time.time(),
                            'ai_confidence': self.current_confidence,
                            'ai_validated': True
                        })
                        self.calibration_count += 1
                        print(f"AI prediction saved with confidence {self.current_confidence:.2f}")
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
                    # Save manual position with correction metadata
                    self.calibration_points.append({
                        'frame_number': self.current_frame_number,
                        'x': x,
                        'y': y,
                        'timestamp': time.time(),
                        'ai_validated': False,
                        'ai_confidence': None,
                        'is_correction': True,
                        'ai_original_prediction': {
                            'x': self.current_ball_center[0] if self.current_ball_center else None,
                            'y': self.current_ball_center[1] if self.current_ball_center else None,
                            'confidence': self.current_confidence if hasattr(self, 'current_confidence') else None
                        }
                    })
                    self.calibration_count += 1
                    self.save_results()
                    self.force_next_frame = True

            except Exception as e:
                print(f"Error handling mouse click: {str(e)}")
                import traceback
                traceback.print_exc()

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
            # Convert numpy/torch types to native Python types
            new_point = {
                'frame_number': int(point['frame_number']),
                'x': int(point['x']),
                'y': int(point['y']),
                'timestamp': float(point['timestamp']),
                'ai_confidence': float(point['ai_confidence']) if point['ai_confidence'] is not None else None,
                'ai_validated': bool(point['ai_validated']),
                'is_correction': not point['ai_validated'],  # Track if this was a manual correction
                'ai_original_prediction': point.get('ai_original_prediction'),  # Store original AI prediction if this was a correction
                'session_id': time.strftime("%Y%m%d_%H%M%S")
            }
            existing_results.append(new_point)
        
        # Save combined results
        with open('ball_calibration.json', 'w') as f:
            json.dump(existing_results, f, indent=4)
        print(f"Results appended to ball_calibration.json (Total entries: {len(existing_results)})")

    def detect_ball(self, frame):
        try:
            if frame is None or frame.size == 0:
                return None, 0.0
            
            # Resize frame for faster processing
            frame = cv2.resize(frame, self.frame_size)
            
            # Convert frame to PyTorch tensor
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            # Get predictions
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            
            # Scale coordinates back to original frame size
            scale_x = frame.shape[1] / self.frame_size[0]
            scale_y = frame.shape[0] / self.frame_size[1]
            
            # Filter for ball detections
            ball_detections = []
            for box, score, label in zip(boxes, scores, labels):
                if label == 37 and score > 0.7:
                    # Scale box coordinates
                    scaled_box = [
                        box[0] * scale_x,
                        box[1] * scale_y,
                        box[2] * scale_x,
                        box[3] * scale_y
                    ]
                    ball_detections.append((scaled_box, score))
            
            if ball_detections:
                box, score = max(ball_detections, key=lambda x: x[1])
                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)
                return (center_x, center_y), score
            
            return None, 0.0
        
        except Exception as e:
            print(f"Error in ball detection: {str(e)}")
            return None, 0.0

    def draw_frame(self, frame):
        display_frame = frame.copy()
        
        # Get frame dimensions
        height, width = display_frame.shape[:2]
        
        # Try automatic ball detection
        ball_center, confidence = self.detect_ball(frame)
        
        if ball_center:
            # Draw suggested point in blue (smaller)
            cv2.circle(display_frame, ball_center, 2, (255, 0, 0), -1)  # Smaller center dot
            cv2.circle(display_frame, ball_center, 10, (255, 0, 0), 1)  # Smaller outer circle
            
            # Move confidence text to top right
            confidence_text = f'AI Confidence: {confidence:.2f}'
            text_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.putText(display_frame, confidence_text,
                       (width - text_size[0] - 10, 30),  # Position from right
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw user's point in green if it exists
        if self.current_point:
            cv2.circle(display_frame, self.current_point, 2, (0, 255, 0), -1)
            cv2.circle(display_frame, self.current_point, 10, (0, 255, 0), 1)
        
        # Draw buttons
        self.draw_buttons(display_frame)
        
        # Show calibration progress and instructions in top right
        calibration_text = f'Calibration: {self.calibration_count + 1}/{self.max_calibrations}'
        text_size = cv2.getTextSize(calibration_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(display_frame, calibration_text,
                   (width - text_size[0] - 10, 60),  # Position from right
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        instruction_text = 'Click to correct AI detection or Skip if no ball visible'
        text_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(display_frame, instruction_text,
                   (width - text_size[0] - 10, 90),  # Position from right
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display_frame

    def calibrate(self):
        print("\nStarting calibration loop...")
        self.force_next_frame = False
        frame_count = 0
        
        while self.calibration_count < self.max_calibrations:
            try:
                frame_count += 1
                print(f"\nProcessing frame {frame_count} (Calibration {self.calibration_count + 1}/{self.max_calibrations})")
                
                frame, frame_number = self.get_random_frame()
                if frame is None:
                    print("Error: Could not get valid frame, retrying...")
                    continue
                
                print(f"Successfully loaded frame {frame_number}")
                self.current_frame = frame.copy()
                self.current_frame_number = frame_number
                self.force_next_frame = False
                
                # Do ball detection once per frame
                self.current_ball_center, self.current_confidence = self.detect_ball(frame)
                if self.current_ball_center and self.current_confidence > 0.7:
                    self.current_point = self.current_ball_center
                    print(f"Auto-detection successful: confidence={self.current_confidence:.3f}")
                else:
                    print("No high-confidence ball detection")
                    self.current_point = None
                
                # Frame display loop
                frame_display_count = 0
                while not self.force_next_frame:
                    try:
                        frame_display_count += 1
                        display_frame = self.draw_frame_cached(self.current_frame)
                        if display_frame is None:
                            print("Error: Failed to create display frame")
                            break
                            
                        cv2.imshow('Ball Calibration', display_frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print("Quit requested by user")
                            self.save_results()
                            return self.calibration_points
                        
                        time.sleep(0.03)
                        
                    except Exception as e:
                        print(f"Error in display loop: {str(e)}")
                        break
                    
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue
        
        print("\nCalibration loop completed")
        cv2.destroyAllWindows()
        self.save_results()
        return self.calibration_points

    def draw_crosshair(self, frame, x, y):
        height, width = frame.shape[:2]
        
        # Check if we're near the ball (either AI detected or manually marked)
        near_ball = False
        ball_pos = None
        
        if self.current_point:
            ball_pos = self.current_point
        elif hasattr(self, 'current_ball_center') and self.current_ball_center:
            ball_pos = self.current_ball_center
        
        if ball_pos:
            dist = np.sqrt((x - ball_pos[0])**2 + (y - ball_pos[1])**2)
            near_ball = dist < 50  # Within 50 pixels of ball
        
        if near_ball:
            # Draw only the center dot when near ball
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        else:
            # Draw full crosshair when not near ball
            gap = 30  # Pixels to leave empty near cursor
            
            # Draw horizontal line with gap
            cv2.line(frame, (0, y), (x - gap, y), (255, 255, 255), 2)
            cv2.line(frame, (x + gap, y), (width, y), (255, 255, 255), 2)
            
            # Draw vertical line with gap
            cv2.line(frame, (x, 0), (x, y - gap), (255, 255, 255), 2)
            cv2.line(frame, (x, y + gap), (x, height), (255, 255, 255), 2)

    def draw_frame_cached(self, frame):
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        
        # Use cached ball detection results
        if hasattr(self, 'current_ball_center') and hasattr(self, 'current_confidence'):
            ball_center = self.current_ball_center
            confidence = self.current_confidence
            
            if ball_center:
                # Draw suggested point in blue (smaller)
                cv2.circle(display_frame, ball_center, 2, (255, 0, 0), -1)
                cv2.circle(display_frame, ball_center, 10, (255, 0, 0), 1)
                
                # Move confidence text to top right
                confidence_text = f'AI Confidence: {confidence:.2f}'
                text_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.putText(display_frame, confidence_text,
                           (width - text_size[0] - 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw user's point in green if it exists
        if self.current_point:
            cv2.circle(display_frame, self.current_point, 2, (0, 255, 0), -1)
            cv2.circle(display_frame, self.current_point, 10, (0, 255, 0), 1)
        
        # Draw buttons
        self.draw_buttons(display_frame)
        
        # Show calibration progress and instructions in top right
        calibration_text = f'Calibration: {self.calibration_count + 1}/{self.max_calibrations}'
        text_size = cv2.getTextSize(calibration_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(display_frame, calibration_text,
                   (width - text_size[0] - 10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        instruction_text = 'Click to correct AI detection or Skip if no ball visible'
        text_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(display_frame, instruction_text,
                   (width - text_size[0] - 10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw crosshair last so it's on top of everything
        self.draw_crosshair(display_frame, self.mouse_x, self.mouse_y)
        
        return display_frame

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