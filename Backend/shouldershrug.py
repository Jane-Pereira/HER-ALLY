import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
from playsound import playsound
import os
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])  # First point
    b = np.array([b.x, b.y])  # Mid point
    c = np.array([c.x, c.y])  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

# Function to detect shoulder shrug and provide feedback
def detect_shoulder_shrug(landmarks):
    # Landmarks for left and right shoulder shrugs
    left_shoulder_angle = calculate_angle(
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    )
    right_shoulder_angle = calculate_angle(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    )

    # Get y-coordinates of shoulders and ears
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    left_ear_y = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y
    right_ear_y = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y

    # Threshold angle and distance for shoulder shrug detection
    shrug_angle_threshold = 30  # degrees
    ear_shoulder_distance_threshold = 0.05  # normalized distance threshold

    # Check if shoulders are shrugged
    left_shrug = (left_shoulder_angle > shrug_angle_threshold) and (left_shoulder_y < left_ear_y - ear_shoulder_distance_threshold)
    right_shrug = (right_shoulder_angle > shrug_angle_threshold) and (right_shoulder_y < right_ear_y - ear_shoulder_distance_threshold)

    return left_shrug, right_shrug

# Function to convert text to speech and play it
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("posture_feedback.mp3")
    playsound("posture_feedback.mp3")
    os.remove("posture_feedback.mp3")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables to manage feedback delay
incorrect_posture_start_time = None
feedback_delay = 3  # seconds

# Variables to count shoulder shrugs
total_shrug_count = 0
left_shoulder_shrugged = False
right_shoulder_shrugged = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find pose landmarks
    results = pose.process(rgb_frame)

    # Draw the pose annotation on the image.
    annotated_frame = frame.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Detect shoulder shrug
        left_shrug, right_shrug = detect_shoulder_shrug(results.pose_landmarks.landmark)

        # Check if both shoulders are shrugging
        if left_shrug and right_shrug:
            if not (left_shoulder_shrugged or right_shoulder_shrugged):
                total_shrug_count += 1
                left_shoulder_shrugged = True
                right_shoulder_shrugged = True
        else:
            left_shoulder_shrugged = False
            right_shoulder_shrugged = False

        # Determine feedback
        if left_shrug and right_shrug:
            feedback = 'Shoulder Shrug Correct'
            color = (0, 255, 0)
            incorrect_posture_start_time = None  # Reset the incorrect posture timer
        else:
            feedback = 'Shoulder Shrug Incorrect'
            color = (0, 0, 255)

            if incorrect_posture_start_time is None:
                incorrect_posture_start_time = time.time()  # Start the timer for incorrect posture
            elif time.time() - incorrect_posture_start_time > feedback_delay:
                text_to_speech('Please correct your posture')
                incorrect_posture_start_time = None  # Reset after providing feedback

        # Display feedback and count
        feedback += f' | Total Shrugs: {total_shrug_count}'
        cv2.putText(annotated_frame, feedback, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Draw circles where the hands should be
        left_hand_position = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * frame.shape[1]),
                              int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * frame.shape[0]))
        right_hand_position = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame.shape[1]),
                               int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame.shape[0]))

        cv2.circle(annotated_frame, left_hand_position, 10, (255, 0, 0), -1)
        cv2.circle(annotated_frame, right_hand_position, 10, (255, 0, 0), -1)

    # Display the annotated frame
    cv2.imshow('Shoulder Shrug Detection', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
