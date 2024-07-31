# HER-ALLY

# Introduction
This project is designed to detect specific exercises using real-time webcam input and provide feedback to users on their posture. The exercises implemented in this project are:

  1. Shoulder Shrug
  2. Bent Arm Forward
  3. Bent Arm Sideways
     
The project uses OpenCV and MediaPipe for pose estimation and provides real-time feedback through text-to-speech and visual annotations.

# Salient Features
Real-time Pose Detection: Uses MediaPipe Pose to detect and track key landmarks on the user's body.
Angle Calculation: Calculates angles between joints to determine the correctness of the exercise.
Feedback Mechanism: Provides visual and audio feedback to guide the user towards correct posture.
Performance Metrics: Measures and reports accuracy, latency, and noise level of the detection system.
User Interface: Displays live video feed with annotations indicating exercise status.
Cross-Platform Compatibility: Compatible with Windows, macOS, and Linux.

# Variables Used
mp_pose: MediaPipe Pose solution for pose detection.
pose: Instance of MediaPipe Pose.
mp_drawing: MediaPipe drawing utilities for visualizing pose landmarks.
feedback_delay: Delay in seconds before providing feedback for incorrect posture.
total_shrug_count, total_correct_count, exercise_count: Counters for the respective exercises.
incorrect_posture_start_time: Timestamp for tracking incorrect posture duration.
left_shoulder_shrugged, right_shoulder_shrugged: Flags for detecting shoulder shrugs.
left_correct, right_correct: Flags for detecting correctness of bent arm exercises.

# Software Requirements
Python 3.6+
OpenCV
MediaPipe
NumPy
gTTS (Google Text-to-Speech)
playsound
Pillow (PIL)

# Hardware Requirements
A computer with a webcam.
Sufficient CPU and memory to handle real-time video processing.

# Compiling and Running the Software
1.Install Dependencies:
    Install the required Python packages using pip:
        pip install opencv-python mediapipe numpy gtts playsound pillow
2.Running the Code:
    Ensure your webcam is connected and run the Python script:
        python exercise_detection.py

# Usage Instructions
1. Shoulder Shrug Detection:
  The system will detect shoulder shrugs and provide feedback on whether the posture is correct or incorrect.
  Correct posture: Shoulders are shrugged upwards.
  Incorrect posture: Shoulders are not properly shrugged.

2. Bent Arm Forward Detection:
  The system will detect bent arm forward exercises.
  Correct posture: Arms are bent at approximately 90 degrees forward.
  Incorrect posture: Arms are not bent at the correct angle.

3.Bent Arm Sideways Detection:
  The system will detect bent arm sideways exercises.
  Correct posture: Arms are bent at approximately 90 degrees sideways.
  Incorrect posture: Arms are not bent at the correct angle.

# Visual and Audio Feedback:
  Visual feedback: The screen will display whether the posture is correct or incorrect.
  Audio feedback: A voice prompt will guide the user to correct their posture if it is incorrect.

# Acknowledgements
  This project uses MediaPipe, a public domain software for pose estimation. MediaPipe can be found at:

