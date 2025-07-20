import cv2
import numpy as np
from ultralytics import YOLO
import pygame

# Initialize traffic light model
model = YOLO("best.pt")

# Initialize sound
pygame.mixer.init()
pygame.mixer.music.load("alert-sound.mp3")  # replace with your sound file

# Open webcam
cap = cv2.VideoCapture(0)

def is_green_dominant(cropped_img):
    """
    Function to determine whether the colors in a bounding box are green dominant
    Used to determine if a traffic light is green
    
    Input: Frame or bounding box to check

    Ouput: Bool value, green dominant or not
    """
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
    
    # Threshold: if more than 10% pixels are green say it's a green light
    return green_ratio > 0.1


"""
Main app loop

Checks each frame for a traffic light, and if it's green. If so, play a sound.
"""
green_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference (suppress verbose for speed)
    results = model(frame, verbose=False)[0]

    # To prevent multiple sounds playing per singular green light
    green_found_this_frame = False

    # Iterate detected boxes and labels
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):

        # Class index for traffic light (In this case 0, 9 for base yolo models)
        if int(cls) == 0:
            x1, y1, x2, y2 = map(int, box)

            # Crop the detected traffic light area
            cropped = frame[y1:y2, x1:x2]

            # Check if green is dominant and play sound
            if is_green_dominant(cropped):

                # Draw green box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                if not pygame.mixer.music.get_busy() and not green_detected:
                    pygame.mixer.music.play()
                    green_detected = True
            else:
                # Draw red box if not green light
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Shows real-time view
    cv2.imshow("Traffic Light Detection", frame)

    # Green light in frame reset
    if not green_found_this_frame:
        green_detected = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
