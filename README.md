# Ball Detection AI Calibration Tool

[Look at the post for more details](https://www.johannesmerwe.com/writing/a-guide-to-writing-your-first-ai)

![Ball Detection AI Calibration Tool](https://www.johannesmerwe.com/_next/image?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2F5szmeqkp%2Fproduction%2F81950e76a10e4b1d860657f4d19c10a32741c1a9-3024x1698.png%3Frect%3D833%2C413%2C1804%2C1242%26fit%3Dmax%26auto%3Dformat&w=3840&q=75)

A Python-based tool for calibrating AI ball detection through user feedback. This project helps improve AI accuracy by collecting human-validated ball positions from video frames.

## Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- OpenCV
- PyTorch
- Torchvision
- NumPy
- PIL (Python Imaging Library)

### Installation

1. Create and activate a virtual environment:

```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install opencv-python numpy pillow
```

For GPU support, visit [PyTorch's installation guide](https://pytorch.org/get-started/locally/) to get the correct installation command for your system.

### Usage

1. Place your video file in the project directory (supports mp4 format)

2. Run the script:

```bash
python main.py
```

2. The program will open a window showing frames from your video.

3. For each frame:
   - If the AI correctly identifies the ball: Click "Confirm Ball"
   - If there's no ball: Click "No Ball Here"
   - If the AI is wrong: Click on the correct ball location
   - To skip the current frame: Click "Next Frame"

4. Results are automatically saved to `ball_calibration.json`

### Key Features

- Random frame selection for diverse training data
- Real-time AI predictions with confidence scores
- User-friendly interface for corrections
- Automatic saving of calibration data
- Progress tracking and accuracy statistics

### Tips

- Press 'q' at any time to quit and save your progress
- The tool will process 5 frames by default (adjustable in the code)
- Higher confidence scores (shown in the top right) indicate more certain predictions
- The JSON file can be used later for model retraining

## File Structure

- `ball_analysis.py` - Main script for running the calibration tool
- `ball_calibration.json` - Stores calibration results and corrections
- `ball_tracker_model.pth` - Saved model state (created after first run)
