import cv2

# Load the video stream from the webcam
cap = cv2.VideoCapture(0)

# Define the dimensions of the rectangle
rect_width = 50
rect_height = 50

# Define the target height for the resized frame
target_height = 600

# Create a trackbar to adjust the threshold value
def nothing(x):
    pass

cv2.namedWindow('Flashlight Detection')
cv2.createTrackbar('Threshold', 'Flashlight Detection', 200, 255, nothing)

while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get the threshold value from the trackbar
    threshold = cv2.getTrackbarPos('Threshold', 'Flashlight Detection')

    # Threshold the grayscale image to create a binary image of the flashlight
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find the contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if len(contours) > 0:
        # Find the contour with the largest area (i.e. the brightest part of the image)
        flashlight_contour = min(contours, key=cv2.contourArea)

        # Calculate the centroid of the flashlight contour
        M = cv2.moments(flashlight_contour)
        if M['m00'] != 0:
            flashlight_x = int(M['m10'] / M['m00'])
            flashlight_y = int(M['m01'] / M['m00'])
            # print(f"Flashlight coordinates: ({flashlight_x}, {flashlight_y})")
            print(f"Flashlight coordinates: ({float(flashlight_x)}, {float(flashlight_y)})")

        else:
            flashlight_x = 0
            flashlight_y = 0

        # Calculate the coordinates of the top-left corner of the rectangle
        rect_x = flashlight_x - rect_width // 2
        rect_y = flashlight_y - rect_height // 2

        # Draw the rectangle on the frame
        frame = cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)

    # Resize the frame to the target height while keeping the aspect ratio
    height, width = frame.shape[:2]
    ratio = target_height / height
    frame = cv2.resize(frame, (int(width * ratio), int(height * ratio)))

    # Display the frame with the rectangle drawn around the flashlight
    cv2.imshow('Flashlight Detection', frame)

    # Wait for a key press to exit
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27 or cv2.getWindowProperty('Flashlight Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()

