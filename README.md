# ASL Sign Language Detector

A real-time American Sign Language (ASL) alphabet detector that recognizes hand signs through your webcam. Built with MediaPipe for hand tracking and a custom PyTorch neural network for classification.

**[Try the Live Demo →](https://aashvij.github.io/Sign-Language-Detector/)**

## Overview

This project detects and classifies ASL alphabet letters in real-time using:
- **MediaPipe Hands** for detecting 21 3D hand landmarks from video frames
- **Fully Connected Neural Network (FCNN)** trained in PyTorch to classify hand poses
- **Web deployment** using TypeScript/JavaScript for browser-based inference

The model achieves **95%+ accuracy** on ASL alphabet recognition.

## How It Works

1. **Hand Detection:** MediaPipe identifies hand landmarks (x, y, z coordinates for 21 points) from each video frame
2. **Feature Extraction:** Landmark coordinates are normalized and flattened into a feature vector
3. **Classification:** The trained FCNN predicts which ASL letter the hand pose represents
4. **Display:** The predicted letter is shown in real-time on the video feed

## Tech Stack

| Component | Technology |
|-----------|------------|
| Hand Tracking | MediaPipe |
| Model Training | PyTorch, NumPy, pandas |
| Data Collection | Python, OpenCV |
| Web App | TypeScript, JavaScript, HTML/CSS |
| Dataset | Kaggle ASL dataset + custom captures |

## Project Structure

```
├── model/                  # Trained PyTorch model
├── data/                   # Training data and preprocessing scripts
├── web/                    # Web application source
│   ├── index.html
│   └── ...
├── train.py                # Model training script
├── collect_data.py         # Data collection using webcam
└── README.md
```

## Running Locally

### Prerequisites
- Python 3.8+
- Node.js (for web app)
- Webcam

### Training the Model

```bash
# Install dependencies
pip install torch mediapipe opencv-python pandas numpy

# Collect custom data (optional)
python collect_data.py

# Train the model
python train.py
```

### Running the Web App

```bash
# Navigate to web directory
cd web

# Open index.html in your browser, or use a local server
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

## What I Learned

- Integrating MediaPipe's hand tracking pipeline into both Python and web environments
- Building and training neural networks with PyTorch
- Processing real-time video streams with OpenCV
- Deploying ML models to the browser

## Future Improvements

- [ ] Expand to full ASL words/phrases, not just alphabet
- [ ] Add support for two-handed signs
- [ ] Improve accuracy on similar-looking letters (e.g., M/N, A/S)
- [ ] Mobile-responsive design

## Resources

- [MediaPipe Hands Documentation](https://google.github.io/mediapipe/solutions/hands.html)
- [My Medium Article on This Project](https://medium.com/@aashvijain.dev/mediapipe-sign-language-detector-00ccc0914988)

---

Built by [Aashvi Jain](https://github.com/aashvij)
