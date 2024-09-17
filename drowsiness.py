import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import RPi.GPIO as GPIO
import time

# GPIO pin configuration
ENA = 4  # PWM pin for speed control
IN1 = 17  # Direction control pin 1
IN2 = 27  # Direction control pin 2
RED_LED_PIN = 22  # Red LED for drowsiness alert
GREEN_LED_PIN = 23  # Green LED for normal state
BUZZER_PIN = 24  # Buzzer for alert

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# PWM setup on ENA pin
pwm = GPIO.PWM(ENA, 100)  # PWM on ENA with 100Hz frequency
pwm.start(0)  # Initially, the motor is off

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define eye landmarks
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

def compute_ear(eye):
    # Compute the distances between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    # Compute the distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    
    # Compute the EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Define thresholds
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20

# Initialize frame counter
counter = 0
motor_running = True  # Initially, the motor is running

# Function to control motor, LEDs, and buzzer
def control_motor_and_alerts(drowsy):
    if drowsy:
        # Stop the motor by setting IN1 and IN2 to LOW and duty cycle to 0
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.LOW)
        pwm.ChangeDutyCycle(0)

        # Turn on the red LED and buzzer, turn off the green LED
        GPIO.output(RED_LED_PIN, GPIO.HIGH)
        GPIO.output(GREEN_LED_PIN, GPIO.LOW)
        GPIO.output(BUZZER_PIN, GPIO.HIGH)

        print("Motor stopped, drowsiness detected: RED LED ON, BUZZER ON.")
    else:
        # Run the motor forward with 70% speed
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        pwm.ChangeDutyCycle(70)

        # Turn on the green LED, turn off the red LED and buzzer
        GPIO.output(RED_LED_PIN, GPIO.LOW)
        GPIO.output(GREEN_LED_PIN, GPIO.HIGH)
        GPIO.output(BUZZER_PIN, GPIO.LOW)

        print("Motor running at 70% speed: GREEN LED ON, system normal.")

# Start capturing video
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = [(int(point.x * w), int(point.y * h)) for point in face_landmarks.landmark]
                
                left_eye = [landmarks[idx] for idx in LEFT_EYE_INDICES]
                right_eye = [landmarks[idx] for idx in RIGHT_EYE_INDICES]
                
                left_ear = compute_ear(left_eye)
                right_ear = compute_ear(right_eye)
                
                ear = (left_ear + right_ear) / 2.0
                
                # Draw eye landmarks
                for (x, y) in left_eye:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                for (x, y) in right_eye:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                # Check if EAR is below the threshold
                if ear < EAR_THRESHOLD:
                    counter += 1
                    if counter >= CONSEC_FRAMES:
                        cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        control_motor_and_alerts(drowsy=True)
                else:
                    counter = 0
                    control_motor_and_alerts(drowsy=False)
        
        cv2.imshow("Drowsiness Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pwm.stop()
    GPIO.cleanup()
