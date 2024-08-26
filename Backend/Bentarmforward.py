import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
from playsound import playsound
import os
import time
from PIL import Image, ImageDraw, ImageFont

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

# Function to detect bent arm forward exercise and provide feedback
def detect_bent_arm_forward(landmarks):
    left_elbow_angle = calculate_angle(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    )
    right_elbow_angle = calculate_angle(
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    )

    correct_angle_threshold = 90  # degrees

    left_correct = np.abs(left_elbow_angle - correct_angle_threshold) < 10
    right_correct = np.abs(right_elbow_angle - correct_angle_threshold) < 10

    return left_correct, right_correct

# Function to convert text to speech and play it
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("posture_feedback.mp3")
    playsound("posture_feedback.mp3")
    os.remove("posture_feedback.mp3")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables to manage feedback delay
frame_count = 0
incorrect_posture_start_time = None
feedback_delay = 6  # seconds

# Variables to count correct exercises
total_correct_count = 0
left_arm_correct = False
right_arm_correct = False
exercise_count = 0
arms_down = True  # To track if arms are down initially

# Load Times New Roman font
try:
    font_path = "/path/to/Times_New_Roman.ttf"  # Provide the correct path to the font file
    font = ImageFont.truetype(font_path, 32)
    large_font = ImageFont.truetype(font_path, 48)
except IOError:

    font = ImageFont.load_default()
    large_font = ImageFont.load_default()

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

        left_correct, right_correct = detect_bent_arm_forward(results.pose_landmarks.landmark)

        # Check if both arms are down
        left_arm_down = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y > results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_arm_down = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y > results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

        if left_correct and right_correct:
            if arms_down:  # Only count if arms were down before
                exercise_count += 1
                arms_down = False
        else:
            if left_arm_down and right_arm_down:
                arms_down = True

        if left_correct and right_correct:
            feedback = 'Bent Arm Forward Correct'
            color = (0, 255, 0)
            incorrect_posture_start_time = None  # Reset the incorrect posture timer
        else:
            feedback = 'Bent Arm Forward Incorrect'
            color = (0, 0, 255)

            if incorrect_posture_start_time is None:
                incorrect_posture_start_time = time.time()  # Start the timer for incorrect posture
            elif time.time() - incorrect_posture_start_time > feedback_delay:
                text_to_speech('Please correct your posture')
                incorrect_posture_start_time = None  # Reset after providing feedback

        cv2.putText(annotated_frame, feedback, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'Count: {exercise_count}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        left_hand_position = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * frame.shape[1]),
                              int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * frame.shape[0]))
        right_hand_position = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame.shape[1]),
                               int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame.shape[0]))

        cv2.circle(annotated_frame, left_hand_position, 10, (255, 0, 0), -1)
        cv2.circle(annotated_frame, right_hand_position, 10, (255, 0, 0), -1)

    # Convert the frame to PIL Image for drawing text with custom font
    pil_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Draw the black box and white text for exercise title
    text = "Exercise 2: Bent Arm Forward"
    text_size = draw.textbbox((0, 0), text, font=large_font)
    draw.rectangle([(10, 10), (10 + text_size[2], 10 + text_size[3])], fill="black")
    draw.text((10, 10), text, font=large_font, fill="white")

    # Convert back to OpenCV image
    annotated_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Display the annotated frame
    cv2.imshow('Bent Arm Forward Detection', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
