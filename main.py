import cv2
import dlib
import numpy as np


# Load the shape predictor 68 dat file
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the face detector
detector = dlib.get_frontal_face_detector()

# Load the webcam
cap = cv2.VideoCapture(0)

def mouth_aspect_ratio(points):
    # Compute the euclidean distances between the horizontal mouth landmarks
    a = np.linalg.norm(points[13] - points[19])
    b = np.linalg.norm(points[14] - points[18])
    c = np.linalg.norm(points[15] - points[17])
    d = np.linalg.norm(points[12] - points[16])

    # Compute the MAR
    mar = (a + b + c) / (3 * d)

    return mar


while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray, 0)

    # Loop over the faces
    for face in faces:
        # Get the landmarks for the face
        landmarks = predictor(gray, face)

        # Convert the landmarks to a numpy array
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Get the mouth landmarks
        mouth = landmarks[48:68]

        # Compute the mouth aspect ratio
        mar = mouth_aspect_ratio(mouth)

        # Draw the mouth landmarks on the frame
        for point in mouth:
            cv2.circle(frame, tuple(point), 2, (0, 0, 255), -1)

            # Display the MAR on the frame
            cv2.putText(frame, f'MAR: {mar:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        # Check if the MAR is below a threshold
        if mar > 0.4:
            cv2.putText(frame, 'DROWSINESS ALERT!', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Close the window
cv2.destroyAllWindows()
