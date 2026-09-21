#!/usr/bin/env python3

# Logan Purkiss and Mopel Kitele
import numpy as np
import cv2

# Get the screen size to set the window sizes accordingly
try:
    from screeninfo import get_monitors
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height
except ImportError:
    print("screeninfo module not found. Using default screen size.")
    screen_width = 1200
    screen_height = 900

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Setup the windows and their sizes
windows = ["Raw", "Thresholded", "Tracked"]

capture_width = 640
capture_height = 480

display_width = screen_width // 3
display_height = screen_height // 3

cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)

# Set size of the square around the click to sample the average color from
square_size = 11
half_square = square_size // 2

# Picked color in HSV format to be set later
picked_color = None

# Center of the object in the previous frame
previous_center = None

# Average the HSV values in a square around a point using median value
def sample_hsv(frame, x, y, square_size):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, w = frame.shape[:2]

    y1 = max(0, y - half_square)
    y2 = min(h, y + half_square + 1)

    x1 = max(0, x - half_square)
    x2 = min(w, x + half_square + 1)

    square = hsv_frame[y1:y2, x1:x2]

    median_color = np.median(square, axis=(0, 1))
    color_spread = np.std(square.astype(np.float32), axis=(0, 1))

    return median_color, color_spread

# Get the center of any contour
def get_contour_center(contour):
    M = cv2.moments(contour)

    if M["m00"] == 0:
        return None

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return (cX, cY)

# Mouse callback function to pick color on click, sets the picked_color and tolerances based on the color spread in the sampled area
def on_mouse_click(event, x, y, flags, param):
    global picked_color, previous_center
    global H_tolerance, S_tolerance, V_tolerance

    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param

        picked_color, color_spread = sample_hsv(
            frame,
            x,
            y,
            square_size
        )

        H_tolerance = int(np.clip(
            8 + 2 * color_spread[0],
            8,
            25
        ))

        S_tolerance = int(np.clip(
            30 + 2 * color_spread[1],
            50,
            100
        ))

        V_tolerance = int(np.clip(
            30 + 2 * color_spread[2],
            50,
            100
        ))

        previous_center = None

        print(f"Picked color (HSV): {picked_color}")
        print(
            f"HSV tolerances: "
            f"H={H_tolerance}, "
            f"S={S_tolerance}, "
            f"V={V_tolerance}"
        )

# Instructions
print("Click on an object to track it.\nPress 'q' to quit.")

# Main loop to capture frames and process them
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #Display webcam
    frame1 = cv2.resize(frame, (display_width, display_height))
    cv2.imshow('Raw', frame1)

    # Set the mouse callback for the raw footage window
    cv2.setMouseCallback('Raw', on_mouse_click, frame1)

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

        # Run opening and closing to remove noise and fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If any contours are found, filter them based on area and find the one closest to the previous center
        if contours:
            valid_contours = [
                contour for contour in contours
                if cv2.contourArea(contour) > 500
            ]

            # If there are valid contours, find the one closest to the previous center or the largest one if no previous center exists
            if valid_contours:
                if previous_center is None:
                    target_contour = max(valid_contours, key=cv2.contourArea)
                else:
                    target_contour = min(
                        valid_contours,
                        key=lambda contour: (
                            (get_contour_center(contour)[0] - previous_center[0]) ** 2
                            + (get_contour_center(contour)[1] - previous_center[1]) ** 2
                        )
                    )

            # Draw a bounding box around the largest contour
            x, y, w, h = cv2.boundingRect(target_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Find the centroid of the largest contour and draw it
            M = cv2.moments(target_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                previous_center = (cX, cY) # Save centroid as previous_center

                print(f"Object location: ({cX}, {cY})")

                cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)
                cv2.putText(frame, f"({cX}, {cY})", (cX + 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the windows
    if picked_color is not None:
        frame2 = cv2.resize(closed, (display_width, display_height))
        cv2.imshow('Thresholded', frame2)
        frame3 = cv2.resize(frame, (display_width, display_height))
        cv2.imshow('Tracked', frame3)
    
    # Position the windows
    cv2.moveWindow('Raw', 0, 0)

    if picked_color is not None:
        cv2.moveWindow('Thresholded', display_width, 0)
        cv2.moveWindow('Tracked', display_width * 2, 0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
