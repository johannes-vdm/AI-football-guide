import cv2
import numpy as np
import random
import json
import time
import torch
from torchvision.models import detection
from torchvision.transforms import functional as F
from PIL import Image
import os

class BallTracker:
    def __init__(self, video_path):
        print("\nInitializing BallTracker...")
        try:
            # Store video path for reopening if needed
            self.video_path = video_path
            
            # Learning parameters
            self.learning_rate = 0.001
            self.training_data = []
            self.min_corrections_for_update = 3
            self.last_training_time = time.time()
            self.training_interval = 10  # seconds between training updates
            self.frame_size = (400, 300)  # Smaller frame size for better performance
            
            # Initialize video capture
            self.init_video_capture()
            
            # Cache for frame processing
            self.current_frame = None
            self.current_frame_tensor = None
            self.current_frame_resized = None
            self.display_frame = None
            
            self.calibration_points = []
            self.current_frame_number = None
            self.current_point = None
            self.calibration_count = 0
            self.max_calibrations = 5
            self.force_next_frame = False
            
            # Redesign button layout
            self.button_ai_correct = {'x': 50, 'y': 30, 'w': 150, 'h': 50}
            self.button_ai_wrong = {'x': 220, 'y': 30, 'w': 150, 'h': 50}
            self.button_skip = {'x': 390, 'y': 30, 'w': 150, 'h': 50}
            
            # Mouse tracking
            self.mouse_x = 0
            self.mouse_y = 0
            
            # Initialize model after all parameters are set
            self.setup_model()
            
            # Try to load existing model
            if self.load_model():
                print("Using previously trained model")
            else:
                print("Starting with fresh model")

        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            self.cleanup()
            raise

    def init_video_capture(self):
        """Initialize or reinitialize video capture"""
        if hasattr(self, 'video') and self.video is not None:
            self.video.release()
        
        self.video = cv2.VideoCapture(self.video_path)
        if not self.video.isOpened():
            raise Exception(f"Error: Could not open video file: {self.video_path}")
        
        # Set video decoder parameters for better stability and performance
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        print(f"Video loaded successfully:")
        print(f"- Total frames: {self.total_frames}")
        print(f"- FPS: {self.fps}")
        print(f"- Duration: {self.total_frames/self.fps:.2f} seconds")

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'video') and self.video is not None:
            self.video.release()
        cv2.destroyAllWindows()

    def setup_model(self):
        print("Loading PyTorch model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        self.model.to(self.device)
        
        # Setup optimizer for online learning
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        print(f"Model loaded on {self.device} with online learning enabled")

    def update_model(self, frame, true_center):
        try:
            # Skip if not enough time has passed since last update
            if time.time() - self.last_training_time < self.training_interval:
                return

            # Prepare training data
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
            
            # Create target box (20x20 box around center point)
            x, y = true_center
            target_box = torch.tensor([[
                x - 10, y - 10,  # x1, y1
                x + 10, y + 10   # x2, y2
            ]]).float().to(self.device)
            
            # Prepare target dictionary
            target = {
                'boxes': target_box,
                'labels': torch.tensor([37]).to(self.device),  # 37 is the ball class
                'scores': torch.tensor([1.0]).to(self.device)
            }
            
            # Training step
            self.model.train()
            self.optimizer.zero_grad()
            
            loss_dict = self.model(image_tensor, [target])
            total_loss = sum(loss for loss in loss_dict.values())
            
            total_loss.backward()
            self.optimizer.step()
            
            print(f"\nModel updated with correction (loss: {total_loss.item():.4f})")
            self.last_training_time = time.time()
            
            # Save model periodically
            self.save_model()
            
        except Exception as e:
            print(f"Error during model update: {str(e)}")
        finally:
            self.model.eval()

    def save_model(self):
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_data': self.training_data
            }, 'ball_tracker_model.pth')
            print("Model state saved")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

    def load_model(self):
        try:
            if os.path.exists('ball_tracker_model.pth'):
                checkpoint = torch.load('ball_tracker_model.pth')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.training_data = checkpoint['training_data']
                print("Loaded previously trained model")
                return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
        return False

    def get_random_frame(self):
        start_time = time.time()
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                # Get a random frame from first 80% of the video to avoid end-of-file issues
                random_frame = random.randint(0, int(self.total_frames * 0.8))
                
                # Reset video capture if needed
                if not self.video.isOpened():
                    print("Video capture was closed, reopening...")
                    reopen_start = time.time()
                    self.video = cv2.VideoCapture(self.video_path)
                    print(f"Reopen time: {(time.time() - reopen_start)*1000:.1f}ms")
                
                # Seek to frame
                seek_start = time.time()
                self.video.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
                print(f"Seek time: {(time.time() - seek_start)*1000:.1f}ms")
                
                # Try to read multiple frames to stabilize decoder
                read_start = time.time()
                for _ in range(5):  # Read a few frames to stabilize
                    ret, frame = self.video.read()
                
                # Now read the actual frame we want
                ret, frame = self.video.read()
                print(f"Frame read time: {(time.time() - read_start)*1000:.1f}ms")
                
                if ret and frame is not None and frame.size > 0:
                    print(f"Total frame loading time: {(time.time() - start_time)*1000:.1f}ms")
                    return frame, random_frame
                
                print(f"Failed to load frame {random_frame}, retrying... (Attempt {attempt + 1})")
                
                # Reset video capture on failure
                self.video.release()
                self.video = cv2.VideoCapture(self.video_path)
                
            except Exception as e:
                print(f"Error loading frame: {str(e)} (Attempt {attempt + 1})")
                # Reset video capture on error
                try:
                    self.video.release()
                    self.video = cv2.VideoCapture(self.video_path)
                except:
                    pass
        
        print("Failed to get valid random frame after multiple attempts")
        print("Trying to read next available frame...")
        
        # If all random attempts fail, try reading sequentially
        try:
            self.video.release()
            self.video = cv2.VideoCapture(self.video_path)
            ret, frame = self.video.read()
            if ret and frame is not None and frame.size > 0:
                current_frame = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
                return frame, current_frame
        except Exception as e:
            print(f"Error reading sequential frame: {str(e)}")
        
        return None, None

    def draw_buttons(self, frame):
        # Button states depend on current situation
        if hasattr(self, 'current_ball_center') and self.current_ball_center:
            # AI detected a ball
            cv2.rectangle(frame, 
                         (self.button_ai_correct['x'], self.button_ai_correct['y']), 
                         (self.button_ai_correct['x'] + self.button_ai_correct['w'], 
                          self.button_ai_correct['y'] + self.button_ai_correct['h']), 
                         (0, 255, 0), -1)
            cv2.putText(frame, 'Confirm Ball', 
                       (self.button_ai_correct['x'] + 20, self.button_ai_correct['y'] + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            cv2.rectangle(frame, 
                         (self.button_ai_wrong['x'], self.button_ai_wrong['y']),
                         (self.button_ai_wrong['x'] + self.button_ai_wrong['w'], 
                          self.button_ai_wrong['y'] + self.button_ai_wrong['h']),
                         (0, 255, 255), -1)
            cv2.putText(frame, 'Wrong/No Ball',
                       (self.button_ai_wrong['x'] + 20, self.button_ai_wrong['y'] + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        else:
            # No ball detected by AI
            cv2.rectangle(frame, 
                         (self.button_ai_correct['x'], self.button_ai_correct['y']), 
                         (self.button_ai_correct['x'] + self.button_ai_correct['w'], 
                          self.button_ai_correct['y'] + self.button_ai_correct['h']), 
                         (0, 255, 0), -1)
            cv2.putText(frame, 'Mark Ball', 
                       (self.button_ai_correct['x'] + 30, self.button_ai_correct['y'] + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            cv2.rectangle(frame, 
                         (self.button_ai_wrong['x'], self.button_ai_wrong['y']),
                         (self.button_ai_wrong['x'] + self.button_ai_wrong['w'], 
                          self.button_ai_wrong['y'] + self.button_ai_wrong['h']),
                         (0, 255, 255), -1)
            cv2.putText(frame, 'No Ball Here',
                       (self.button_ai_wrong['x'] + 20, self.button_ai_wrong['y'] + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Skip button is always the same
        cv2.rectangle(frame, 
                     (self.button_skip['x'], self.button_skip['y']),
                     (self.button_skip['x'] + self.button_skip['w'], 
                      self.button_skip['y'] + self.button_skip['h']),
                     (0, 0, 255), -1)
        cv2.putText(frame, 'Next Frame',
                   (self.button_skip['x'] + 20, self.button_skip['y'] + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    def draw_point(self, frame):
        if self.current_point:
            cv2.circle(frame, self.current_point, 2, (0, 255, 0), -1)  # Smaller center dot
            cv2.circle(frame, self.current_point, 10, (0, 255, 0), 1)  # Smaller outer circle

    def mouse_callback(self, event, x, y, flags, param):
        # Update mouse position for crosshair
        self.mouse_x = x
        self.mouse_y = y
        
        # Only handle left button clicks
        if event != cv2.EVENT_LBUTTONDOWN:
            return
            
        click_start = time.time()
        try:
            print(f"\nMouse click at (x:{x}, y:{y})")
            
            # Ensure we have a valid frame
            if self.current_frame is None:
                print("No valid frame to process click")
                return
            
            # Check buttons first
            if y <= 80:  # Button area
                if (self.button_ai_correct['x'] <= x <= self.button_ai_correct['x'] + self.button_ai_correct['w'] and
                    self.button_ai_correct['y'] <= y <= self.button_ai_correct['y'] + self.button_ai_correct['h']):
                    print("AI Correct button clicked")
                    if hasattr(self, 'current_ball_center') and self.current_ball_center:
                        # Confirm AI detection
                        save_start = time.time()
                        print("Ball position confirmed!")
                        self.calibration_points.append({
                            'frame_number': self.current_frame_number,
                            'x': self.current_ball_center[0],
                            'y': self.current_ball_center[1],
                            'timestamp': time.time(),
                            'ai_confidence': self.current_confidence,
                            'ai_validated': True,
                            'is_correction': False
                        })
                        self.calibration_count += 1
                        print(f"AI prediction saved with confidence {self.current_confidence:.2f}")
                        self.save_results()
                        print(f"Save time: {(time.time() - save_start)*1000:.1f}ms")
                        self.force_next_frame = True
                    else:
                        # Manual ball marking mode
                        print("Please click on the ball location")
                        self.current_point = None
                    return

                elif (self.button_ai_wrong['x'] <= x <= self.button_ai_wrong['x'] + self.button_ai_wrong['w'] and
                      self.button_ai_wrong['y'] <= y <= self.button_ai_wrong['y'] + self.button_ai_wrong['h']):
                    print("AI Wrong button clicked")
                    # No ball in frame
                    save_start = time.time()
                    print("No ball in frame")
                    self.calibration_points.append({
                        'frame_number': self.current_frame_number,
                        'x': None,
                        'y': None,
                        'timestamp': time.time(),
                        'ai_validated': False,
                        'ai_confidence': None,
                        'is_correction': True,
                        'no_ball': True,
                        'ai_original_prediction': {
                            'x': self.current_ball_center[0] if hasattr(self, 'current_ball_center') and self.current_ball_center else None,
                            'y': self.current_ball_center[1] if hasattr(self, 'current_ball_center') and self.current_ball_center else None,
                            'confidence': self.current_confidence if hasattr(self, 'current_confidence') else None
                        }
                    })
                    self.calibration_count += 1
                    self.save_results()
                    print(f"Save time: {(time.time() - save_start)*1000:.1f}ms")
                    self.force_next_frame = True
                    return

                elif (self.button_skip['x'] <= x <= self.button_skip['x'] + self.button_skip['w'] and
                      self.button_skip['y'] <= y <= self.button_skip['y'] + self.button_skip['h']):
                    print("Skip button clicked")
                    print("Moving to next frame...")
                    self.force_next_frame = True
                    return

            # Handle manual ball position click
            else:
                print("Manual ball position click")
                if not hasattr(self, 'current_ball_center') or not self.current_ball_center:
                    # First click in manual mode
                    save_start = time.time()
                    print(f"Ball position marked at {x}, {y}")
                    self.current_point = (x, y)
                    self.calibration_points.append({
                        'frame_number': self.current_frame_number,
                        'x': x,
                        'y': y,
                        'timestamp': time.time(),
                        'ai_validated': False,
                        'ai_confidence': None,
                        'is_correction': True,
                        'no_ball': False,
                        'ai_original_prediction': None
                    })
                    self.calibration_count += 1
                    self.save_results()
                    print(f"Save time: {(time.time() - save_start)*1000:.1f}ms")
                    self.force_next_frame = True
                else:
                    # Correcting AI detection
                    save_start = time.time()
                    print(f"Correcting ball position to {x}, {y}")
                    self.current_point = (x, y)
                    self.calibration_points.append({
                        'frame_number': self.current_frame_number,
                        'x': x,
                        'y': y,
                        'timestamp': time.time(),
                        'ai_validated': False,
                        'ai_confidence': None,
                        'is_correction': True,
                        'no_ball': False,
                        'ai_original_prediction': {
                            'x': self.current_ball_center[0],
                            'y': self.current_ball_center[1],
                            'confidence': self.current_confidence
                        }
                    })
                    self.calibration_count += 1
                    self.save_results()
                    print(f"Save time: {(time.time() - save_start)*1000:.1f}ms")
                    self.force_next_frame = True
                    
                    # Update model with correction
                    model_start = time.time()
                    if self.current_frame is not None:
                        self.update_model(self.current_frame, (x, y))
                    print(f"Model update time: {(time.time() - model_start)*1000:.1f}ms")

        except Exception as e:
            print(f"Error handling mouse click: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"Total click handling time: {(time.time() - click_start)*1000:.1f}ms")
            # Ensure window is updated
            cv2.waitKey(1)

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
                'x': float(point['x']) if point.get('x') is not None else None,
                'y': float(point['y']) if point.get('y') is not None else None,
                'timestamp': float(point['timestamp']),
                'ai_confidence': float(point['ai_confidence']) if point.get('ai_confidence') is not None else None,
                'ai_validated': bool(point['ai_validated']),
                'is_correction': bool(point.get('is_correction', False)),
                'no_ball': bool(point.get('no_ball', False)),
                'session_id': time.strftime("%Y%m%d_%H%M%S")
            }
            
            # Handle AI original prediction if it exists
            if point.get('ai_original_prediction'):
                orig = point['ai_original_prediction']
                new_point['ai_original_prediction'] = {
                    'x': float(orig['x']) if orig.get('x') is not None else None,
                    'y': float(orig['y']) if orig.get('y') is not None else None,
                    'confidence': float(orig['confidence']) if orig.get('confidence') is not None else None
                }
            else:
                new_point['ai_original_prediction'] = None
            
            existing_results.append(new_point)
        
        # Save combined results
        with open('ball_calibration.json', 'w') as f:
            json.dump(existing_results, f, indent=4)
        print(f"Results appended to ball_calibration.json (Total entries: {len(existing_results)})")
        
        # Clear calibration points after saving
        self.calibration_points = []

    def detect_ball(self, frame):
        start_time = time.time()
        try:
            if frame is None or frame.size == 0:
                return None, 0.0
            
            # Get original frame dimensions
            orig_height, orig_width = frame.shape[:2]
            
            # Reuse cached resized frame if possible
            resize_start = time.time()
            if self.current_frame_resized is None:
                self.current_frame_resized = cv2.resize(frame, self.frame_size)
            resize_time = time.time() - resize_start
            
            # Reuse cached tensor if possible
            tensor_start = time.time()
            if self.current_frame_tensor is None:
                image = Image.fromarray(cv2.cvtColor(self.current_frame_resized, cv2.COLOR_BGR2RGB))
                self.current_frame_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
            tensor_time = time.time() - tensor_start
            
            # Model inference
            inference_start = time.time()
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(self.current_frame_tensor)
            inference_time = time.time() - inference_start
            
            # Process predictions
            post_start = time.time()
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            
            # Calculate scale factors
            scale_x = orig_width / self.frame_size[0]
            scale_y = orig_height / self.frame_size[1]
            
            # Filter for ball detections
            ball_detections = []
            for box, score, label in zip(boxes, scores, labels):
                if label == 37 and score > 0.3:
                    scaled_box = [
                        int(box[0] * scale_x),
                        int(box[1] * scale_y),
                        int(box[2] * scale_x),
                        int(box[3] * scale_y)
                    ]
                    ball_detections.append((scaled_box, score))
            post_time = time.time() - post_start
            
            total_time = time.time() - start_time
            print(f"\nBall Detection Pipeline:")
            print(f"- Resize: {resize_time*1000:.1f}ms")
            print(f"- Tensor conversion: {tensor_time*1000:.1f}ms")
            print(f"- Model inference: {inference_time*1000:.1f}ms")
            print(f"- Post-processing: {post_time*1000:.1f}ms")
            print(f"- Total time: {total_time*1000:.1f}ms")
            
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

    def draw_crosshair(self, frame, x, y):
        height, width = frame.shape[:2]
        gap_size = 30  # 30 pixel gap in the middle
        
        # If we have a ball detection, draw crosshair on the ball
        if hasattr(self, 'current_ball_center') and self.current_ball_center:
            x, y = self.current_ball_center
            
            # Draw vertical line with gap
            cv2.line(frame, (x, 0), (x, y - gap_size), (255, 255, 255), 1)  # Top part
            cv2.line(frame, (x, y + gap_size), (x, height), (255, 255, 255), 1)  # Bottom part
            
            # Draw horizontal line with gap
            cv2.line(frame, (0, y), (x - gap_size, y), (255, 255, 255), 1)  # Left part
            cv2.line(frame, (x + gap_size, y), (width, y), (255, 255, 255), 1)  # Right part
            
            # Draw green dot at intersection
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Green dot

    def draw_frame_cached(self, frame):
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        
        # Draw buttons first
        self.draw_buttons(display_frame)
        
        # Draw ball detection
        if hasattr(self, 'current_ball_center') and hasattr(self, 'current_confidence'):
            ball_center = self.current_ball_center
            confidence = self.current_confidence
            
            if ball_center:
                # Draw AI ball detection (blue)
                cv2.circle(display_frame, ball_center, 3, (255, 255, 255), -1)  # White background
                cv2.circle(display_frame, ball_center, 2, (255, 0, 0), -1)      # Blue center
                cv2.circle(display_frame, ball_center, 20, (255, 0, 0), 2)      # Blue outer circle
                
                # Show confidence
                confidence_text = f'AI Confidence: {confidence:.2f}'
                text_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.putText(display_frame, confidence_text,
                           (width - text_size[0] - 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw user's point in green if it exists
        if self.current_point:
            cv2.circle(display_frame, self.current_point, 3, (255, 255, 255), -1)  # White background
            cv2.circle(display_frame, self.current_point, 2, (0, 255, 0), -1)      # Green center
            cv2.circle(display_frame, self.current_point, 20, (0, 255, 0), 2)      # Green outer circle
        
        # Show calibration progress and instructions
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
        
        # Draw crosshair last so it's always on top
        self.draw_crosshair(display_frame, self.mouse_x, self.mouse_y)
        
        return display_frame

    def calculate_accuracy(self, calibration_points):
        total_frames = len(calibration_points)
        correct_predictions = 0
        avg_confidence = 0
        total_with_confidence = 0
        
        for point in calibration_points:
            if point.get('ai_validated', False):
                # AI was correct and user confirmed
                correct_predictions += 1
                avg_confidence += point.get('ai_confidence', 0)
                total_with_confidence += 1
            elif point.get('no_ball', False):
                # AI made a prediction when there was no ball
                if point.get('ai_original_prediction') is None:
                    # AI correctly didn't detect a ball
                    correct_predictions += 1
            elif point.get('is_correction', False) and point.get('ai_original_prediction'):
                # User had to correct AI's prediction
                orig = point['ai_original_prediction']
                if orig.get('confidence'):
                    avg_confidence += orig['confidence']
                    total_with_confidence += 1
        
        accuracy = (correct_predictions / total_frames) * 100 if total_frames > 0 else 0
        avg_confidence = (avg_confidence / total_with_confidence) * 100 if total_with_confidence > 0 else 0
        
        return accuracy, avg_confidence

    def calibrate(self):
        print("\nStarting calibration loop...")
        self.force_next_frame = False
        frame_count = 0
        
        while self.calibration_count < self.max_calibrations:
            try:
                frame_count += 1
                print(f"\nProcessing frame {frame_count} (Calibration {self.calibration_count + 1}/{self.max_calibrations})")
                
                # Load frame
                frame, frame_number = self.get_random_frame()
                if frame is None or frame.size == 0:
                    print("Error: Could not get valid frame, retrying...")
                    continue
                
                print(f"Successfully loaded frame {frame_number}")
                
                # Reset all state for new frame
                self.current_frame = frame.copy()
                self.current_frame_number = frame_number
                self.current_frame_tensor = None
                self.current_frame_resized = None
                self.display_frame = None
                self.current_point = None
                self.force_next_frame = False
                self.current_ball_center = None
                self.current_confidence = 0.0
                
                # Ball detection
                self.current_ball_center, self.current_confidence = self.detect_ball(frame)
                if self.current_ball_center:
                    print(f"Ball detected at {self.current_ball_center} with confidence {self.current_confidence:.2f}")
                else:
                    print("No ball detected")
                    self.current_ball_center = None
                    self.current_confidence = 0.0
                
                # Frame display loop
                last_draw_time = time.time()
                frame_shown = False
                
                while not self.force_next_frame:
                    try:
                        current_time = time.time()
                        if current_time - last_draw_time < 0.033:  # Limit to 30 FPS
                            cv2.waitKey(1)  # Keep processing events
                            continue
                        
                        # Validate current frame
                        if self.current_frame is None or self.current_frame.size == 0:
                            print("Error: Invalid frame in display loop")
                            break
                        
                        # Always redraw the frame to ensure fresh state
                        self.display_frame = self.draw_frame_cached(self.current_frame)
                        
                        if self.display_frame is None:
                            print("Error: Failed to create display frame")
                            break
                            
                        cv2.imshow('Ball Calibration', self.display_frame)
                        last_draw_time = current_time
                        frame_shown = True
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print("Quit requested by user")
                            self.save_results()
                            return self.calibration_points
                        
                    except Exception as e:
                        print(f"Error in display loop: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        break
                    
                # If frame was never shown successfully, retry
                if not frame_shown:
                    print("Frame was never shown successfully, retrying...")
                    continue
                    
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
            
            # Wait for any pending saves to complete before moving to next frame
            cv2.waitKey(1)
            
            # Reset display frame to force redraw on next frame
            self.display_frame = None
        
        print("\nCalibration loop completed")
        cv2.destroyAllWindows()
        self.save_results()
        
        # Calculate and display accuracy
        accuracy, avg_confidence = self.calculate_accuracy(self.calibration_points)
        print("\nCalibration Results:")
        print(f"AI Accuracy: {accuracy:.1f}%")
        print(f"Average Confidence: {avg_confidence:.1f}%")
        print(f"Total Frames Processed: {frame_count}")
        
        return self.calibration_points

    def start(self):
        try:
            print("Starting calibration mode...")
            print("Instructions:")
            print("1. Click on the ball in the frame")
            print("2. Click 'AI Correct' to save the position")
            print("3. Click 'AI Wrong' if ball is not visible")
            print("4. Click 'Skip' if ball is not visible")
            print("5. Repeat for 5 different frames")
            print("Press 'q' to quit at any time")
            print("Results will be saved to ball_calibration.json")
            
            # Create window and set mouse callback
            cv2.namedWindow('Ball Calibration')
            cv2.setMouseCallback('Ball Calibration', self.mouse_callback)
            
            # Reset state
            self.calibration_count = 0
            self.calibration_points = []
            self.current_frame = None
            self.current_frame_number = None
            self.current_point = None
            self.force_next_frame = False
            
            # Reinitialize video capture
            self.init_video_capture()
            
            calibration_data = self.calibrate()
            
            if calibration_data:
                print("\nCalibration completed!")
                print(f"Collected {len(calibration_data)} ball positions")
            
        except Exception as e:
            print(f"Error during calibration: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

if __name__ == "__main__":
    tracker = BallTracker("barrea.mp4")
    tracker.start() 