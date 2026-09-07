#!/usr/bin/env python3
import numpy as np
import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Setup the windows and their sizes
windows = ["Raw", "Thresholded", "Tracked"]
win_width = 640
win_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, win_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, win_height)

# Set size of the square around the click to sample the average color from
square_size = 11
half_square = square_size // 2

# Tolerances for thresholding
H_tolerance = 10
S_tolerance = 60
V_tolerance = 60

# Picked color in HSV format to be set later
picked_color = None

# Average the HSV values in a square around a point
def sample_average_hsv(frame, x, y, square_size):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    y1, y2 = max(0, y - half_square), min(h, y + half_square + 1)
    x1, x2 = max(0, x - half_square), min(w, x + half_square + 1)
    square = hsv_frame[y1:y2, x1:x2]
    return square.mean(axis=(0, 1))

# Mouse callback function to pick color on click
def on_mouse_click(event, x, y, flags, param):
    global picked_color
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param
        picked_color = sample_average_hsv(frame, x, y, square_size)
        print(f"Picked color (HSV): {picked_color}")

# Main loop to capture frames and process them
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Set the mouse callback for the raw footage window
    cv2.setMouseCallback('Raw', on_mouse_click, frame)

    # If a color has been picked, process the frame to find that color
    if picked_color is not None:
        # Convert the frame to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for thresholding
        lower_bound = np.array([
            max(0, picked_color[0] - H_tolerance), 
            max(0, picked_color[1] - S_tolerance), 
            max(0, picked_color[2] - V_tolerance)])
        upper_bound = np.array([
            min(179, picked_color[0] + H_tolerance), 
            min(255, picked_color[1] + S_tolerance), 
            min(255, picked_color[2] + V_tolerance)])

        # Threshold the HSV image to get only the colors in the range
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # Bitwise-AND mask and original image
        threshold_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Display the windows
    frame1 = cv2.resize(frame, (win_width, win_height))
    if picked_color is not None:
        frame2 = cv2.resize(mask, (win_width, win_height))
        cv2.imshow('Thresholded', frame2)
    frame3 = cv2.resize(frame, (win_width, win_height))
    
    cv2.imshow('Raw', frame1)
    cv2.imshow('Tracked', frame3)

    cv2.moveWindow('Raw', 0, 0)
    cv2.moveWindow('Thresholded', win_width, 0)
    cv2.moveWindow('Tracked', win_width * 2, 0)

    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()