import cv2
import numpy as np
from ultralytics import YOLO
import pygame

# Initialize YOLOv8 with traffic light model
model = YOLO("best.pt")

# Initialize pygame mixer for sound
pygame.mixer.init()
pygame.mixer.music.load("alert-sound.mp3")  # replace with your sound file

# Open webcam
cap = cv2.VideoCapture(0)

def is_green_dominant(cropped_img):
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    
    # Define green color range in HSV
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    
    # Create mask to extract green areas
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    green_pixels = cv2.countNonZero(mask)
    total_pixels = cropped_img.shape[0] * cropped_img.shape[1]
    
    green_ratio = green_pixels / total_pixels
    
    # Threshold: if more than 20% pixels are green, say it's green dominant
    return green_ratio > 0.2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference (suppress verbose for speed)
    results = model(frame, verbose=False)[0]

    # Iterate detected boxes and labels
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        # Assuming class index for traffic light is known (e.g., 9 in COCO)
        # Adjust this if you trained your own classes
        if int(cls) == 9:
            x1, y1, x2, y2 = map(int, box)

            # Crop the detected traffic light area
            cropped = frame[y1:y2, x1:x2]

            # Check if green is dominant in this crop
            if is_green_dominant(cropped):
                # Draw box green and play sound
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
            else:
                # Draw box red if not green light
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.imshow("Traffic Light Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
