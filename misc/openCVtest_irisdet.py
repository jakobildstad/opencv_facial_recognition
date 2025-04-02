import numpy as np
import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error: Could not open camera.")
    exit()

#Video loop
while True:
    ret, frame = capture.read()

    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Assume a face region has been detected; x, y, w, h are the face bounding box coordinates
        # Extract the eye/iris region from the grayscale image (you might want to adjust the ROI as needed)
        gray_roi = gray[y:y+h, x:x+w]

        # Use HoughCircles to detect circles in the ROI
        circles = cv2.HoughCircles(
            gray_roi,
            cv2.HOUGH_GRADIENT,
            dp=1,  # The inverse ratio of resolution
            minDist=gray_roi.shape[0] / 8,  # Minimum distance between detected circles
            param1=50,  # Higher threshold for the Canny edge detector
            param2=30,  # Threshold for center detection in HoughCircles
            minRadius=20,  # Minimum radius to consider (tune this)
            maxRadius=50   # Maximum radius to consider (tune this)
        )

        # If circles were detected, process them
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center_x, center_y, radius = circle
                # Since gray_roi is a sub-image of the full frame, add the face ROI offsets (x, y) to the circle coordinates
                global_center = (center_x + x, center_y + y)
                # Draw the outer circle on the original frame
                cv2.circle(frame, global_center, radius, (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(frame, global_center, 2, (0, 0, 255), 3)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


capture.release()
cv2.destroyAllWindows()