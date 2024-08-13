# Project Overview
As a college student diving into applications of AI for the first time, I built a web app that detects American Sign Language (ASL) in real-time using numerous technologies. This project was an educational journey through the deep learning, computer vision, and full-stack development.

Project link: - 

## Tech Stack

Frontend: TypeScript, JavaScript, HTML

Backend: Python

AI/ML: MediaPipe, PyTorch, OpenCV

Data: Kaggle datasets

# Key Development Stages

**Implementing MediaPipe**: Integrated Google's MediaPipe API to detect hand landmarks and connections in a live video feed, bringing this functionality to life in a web environment.

![](https://github.com/Sign-Language-Detector/landmarks-animated.gif)

**Building the Neural Network**: Constructed a Fully Connected Neural Network (FCNN) model using PyTorch, using pandas and numpy for efficient data manipulation.

IMG

**Data Collection and Preprocessing**: Deployed MediaPipe with Python and OpenCV to capture and process video frames, creating a quality dataset of 15 test images for each ASL letter.

**Training and Optimization**: Utilized Kaggle dataset to train the model, repeatedly refining the architecture and hyperparameters until achieving 95%+ accuracy.

IMG

**Web Integration**: Exported the trained model as an ONNX file and integrated it into the original web app, creating a responsive and 95% accurate ASL detection system.

# Overcoming Challenges
This project was my first try at building and training a neural network from scratch so it presented several exciting challenges:

**MediaPipe Integration**: While a large quota of documentation exists for OpenCV and Python implementations, resources for JavaScript integration were scarce. This encouraged me to scour the MediaPipe documentation and experiment with various implementation strategies.

**Boosting Accuracy**: My inital training epochs yielded a mere 14% accuracy. Through the repeated refinement and significantly expanding the dataset from 5,000 to 143,000 images (approximately 5,500 per sign), I was able to achieve 96% accuracy.

## Demo
Click here to test out my project yourself! LINK

This demo showcases the web app's ability to recognize and interpret ASL signs in real-time, demonstrating the seamless integration of MediaPipe's hand tracking with my trained neural network.

This project helped enhance my skills in AI and web development and learn more about the large range of applications of computer science. Building upon this project further has the potential to make communication more accessible for the deaf and hard of hearing community.
