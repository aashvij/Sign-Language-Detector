#CODE TO TAKE YOUR OWN PICTURES FOR TRAINING AND CONVERT LANDMARK COORDINATES INTO CSV FILE

import cv2
import os
import time
import uuid
import mediapipe as mp
import csv

# path to images
#IMAGES_PATH = "./practiceImgs"
IMAGES_PATH = "./testingImgs"
IMAGES_PATH = "./asl-alphabet-test"

# all labels
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
#labels = ['b', 'd', 'f', 'k']

numberOfImgs = 15 # for image collection (if no dataset already)

#csv where data should be stored
#outputFile = 'hand_landmarks.csv'
outputFile = 'testingData2.csv'

handsmp = mp.solutions.hands
hands = handsmp.Hands(
    static_image_mode=True,
    max_num_hands = 2,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.6,)
drawingmp = mp.solutions.drawing_utils

def saveData():
    with open(outputFile, mode="w", newline='') as file:
        writer = csv.writer(file)
        
        # header = ['label'] + [f'{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
        # only x and y points
        header = ['label'] + [f'{i}_{axis}' for i in range(21) for axis in ['x', 'y']]

        writer.writerow(header)

        for label in labels:
            folderPath = os.path.join(IMAGES_PATH, label)
            for img_file in os.listdir(folderPath):
                if img_file.endswith(".jpg") or img_file.endswith(".jpeg"):
                    image_path = os.path.join(folderPath, img_file)
                    image = cv2.imread(image_path)
                    
                    # Check if the image is loaded correctly
                    if image is None:
                        print(f"Error reading {image_path}")
                        continue
                    
                    # Convert the BGR image to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Process the image to find hand landmarks
                    results = hands.process(image_rgb)

                    # Extract and save landmark values
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            landmarks = []
                            for landmark in hand_landmarks.landmark:
                                landmarks.append(landmark.x)
                                landmarks.append(landmark.y)
                                #landmarks.append(landmark.z)
                            writer.writerow([label] + landmarks)

            print(f"Landmark data saved to {outputFile}, {label}")


# For collecting images on your own with opencv video capture
def imgCollection():
    cap = cv2.VideoCapture(0)
    for label in labels:
        print("Collecting images for " + label)
        time.sleep(5)

        for imgNum in range(numberOfImgs):
            ret, image = cap.read()
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawingmp.draw_landmarks(
                        image,
                        hand_landmarks,
                        handsmp.HAND_CONNECTIONS
                    )
            
            imageLabel = os.path.join(IMAGES_PATH, label, label + '.' + '{}.jpg'.format(str(uuid.uuid1())))
            cv2.imwrite(imageLabel, image)
            print(f'Saved: {imageLabel}')

            cv2.imshow("pic", image)
            time.sleep(2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def fileCorruptionCheck():
    image_dir = './practiceImgs'

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = cv2.imread(file_path)
                if img is None:
                    print(f"Corrupted image file: {file_path}")
                    os.remove(file_path)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                os.remove(file_path)

saveData()