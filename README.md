# Sign Language Number Recognition

This project uses computer vision and machine learning to recognize hand signs for numbers (0-9) in real-time through a webcam.

## Features

- Real-time sign language number detection using webcam
- Hand tracking with MediaPipe
- Random Forest classifier for number recognition
- Data augmentation for improved model accuracy
- Model evaluation and visualization tools


## Installation

### Prerequisites

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- scikit-learn
- Matplotlib

### Setup

1. Clone this repository:

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your dataset: (or use the available dataset)
   - Create a folder structure like:
   ```
   dataset/
   ├── 0/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── 1/
   │   ├── image1.jpg
   │   └── ...
   └── ...
   ```
   - Each subfolder (0-9) should contain images of hand signs for that number

## Usage

Run the main script:
```bash
python main.py
```

### Controls
- **S**: Take a snapshot and save
- **Q**: Quit the application

## Model Training

You can train your own model using the built-in menu:
1. Run the application
2. Select option 1 to train a new model
3. The model will be saved as `number_sign_model.pkl`

## Project Structure

```
├── main.py     # Main application file
├── HandTrackingModule.py        # Hand detection module
└── dataset/                # Training images 
```
