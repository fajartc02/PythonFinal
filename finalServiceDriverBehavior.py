import cv2
import mediapipe as mp
from imutils import face_utils
import numpy as np
import time
import dlib
import math
from phone import poseDetector
# import vlc
import webbrowser
import base64
import asyncio
from websockets.sync.client import connect
import datetime
import json

class SocketTrigger:
    def save_image(image, event, target = ''):

        print('Reconnecting')
        with connect("ws://103.190.28.211:3100?vehicle_id=MHKA6GJ3JRJ043647&device=jetson") as websocket:
        # with connect("ws://192.168.1.102:3100?vehicle_id=MHKA6GJ3JRJ043647&device=jetson") as websocket:
            cv2.imwrite("frame%d.jpg" % 1, image) 
            with open("frame1.jpg", "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
                jsonData = {
                    "event" : "UPLOAD_IMAGE",
                    "vehicle_id" : "123",
                    "target" : target,
                    "data" : {
                        "message": "Image Upload",
                        "image": "%s"%encoded_string,
                        "behavior_type" : event
                    }
                }
                print(jsonData)
                websocket.send(json.dumps(jsonData))
                message = websocket.recv()
                print(f"Received: {message}")

# Constants for drowsiness detection
EYE_ASPECT_RATIO_TRESHOLD = 0.30
EYE_ASPECT_RATIO_CONSEC_FRAMES = 5
MOUTH_ASPECT_RATIO_TRESHOLD = 50
MOUTH_ASPECT_RATIO_CONSEC_FRAMES = 15

COUNTER_EYE = 0
COUNTER_MOUTH = 0

IS_MOUTH_OPEN_15 = False
IS_EYE_CLOSE = False
IS_DROWSINESS = False

map_flag = 1
map_counter = 0
flag = 0
play_sound = False
drowsy_detected = False
last_alert_time = 0
alert_delay = 1
alert_count = 0


# Initialize MediaPipe modules (Driver Distraction)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Initialize dlib face detector and shape predictor (Drowsy)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../assets/shape_predictor_68_face_landmarks.dat')
face_cascade = cv2.CascadeClassifier('../assets/haarcascade_frontalface_default.xml')
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']


# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    # print(ear)
    return ear

def yawn_aspect_ratio(mouth):
    # pointing mouth 
    distYawn = math.sqrt((math.pow(mouth[10][0] - mouth[2][0], 2) + math.pow(mouth[10][1] - mouth[2][1], 2)))
    # print(distYawn)
    return distYawn 

# Initialize video capture
cap = cv2.VideoCapture(0)

# Get the width and height of the frame from the video capture
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Initialize pose detector
pose_detector = poseDetector()

start_time = time.time()
init_time = datetime.datetime.now()

# Extract the seconds
init_minute = init_time.minute - 1

while cap.isOpened():
    success, frame = cap.read()
    image = frame
    if not success:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the color space from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # To improve performance
    rgb_frame.flags.writeable = False

    # Get the result from MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)

    # To improve performance
    rgb_frame.flags.writeable = True

    # Convert the color space from RGB to BGR
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = frame.shape

    img = pose_detector.findPose(frame)
    lmList = pose_detector.findPosition(frame, draw=False)

    if len(lmList) > 0:
        for lm in lmList:
            cv2.putText(frame, str(lm[0]), (lm[1], lm[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if len(lmList) > 16:  # Ensure enough landmarks are detected
            hand_landmark = lmList[16][1:]  # Hand landmark
            eye_landmark = lmList[8][1:]  # Eye landmark

            # Calculate the distance only if both hand and eye landmarks are detected
            if hand_landmark and eye_landmark:
                distance = pose_detector.calculate_distance(hand_landmark, eye_landmark)
                distance_threshold = 170  # Adjust this threshold as needed

                if distance < distance_threshold:
                    cv2.putText(img, "Making a phone call", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                    SocketTrigger.save_image(image, 'DISTRACTION')
                    # alert.play()

                # Calculate the position of the text based on the original image size
                h, w, _ = img.shape
                text_position = (20, h - 50)

 # Draw circles around the hand and eye landmarks
    if len(lmList) > 16:  # Ensure enough landmarks are detected
        cv2.circle(img, (lmList[16][1], lmList[16][2]), 10, (0, 255, 0), thickness=-1)  # Hand
    if len(lmList) > 8:
        cv2.circle(img, (lmList[8][1], lmList[8][2]), 10, (0, 0, 255), thickness=-1)  # Eye

    cTime = time.time()
    pTime = 0
    fps = 1 / (cTime - pTime)

    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_3d = []
            face_2d = []

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

            # Convert to NumPy arrays
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x_angle = angles[0] * 360
            y_angle = angles[1] * 360
            is_behavior_danger = False
            # See where the user's head is tilting

            # Get the current time including seconds
            current_time = datetime.datetime.now()

            # Extract the seconds
            minuteCount = current_time.minute

            
            
            if y_angle < -10:
                head_pose_text = "Looking Left"
                is_behavior_danger = True
            elif y_angle > 10:
                head_pose_text = "Looking Right"
                is_behavior_danger = True
            elif x_angle < -10:
                head_pose_text = "Looking Down"
                is_behavior_danger = True
            elif x_angle > 10:
                head_pose_text = "Looking Up"
                is_behavior_danger = True
            else:
                head_pose_text = "Forward"
                is_behavior_danger = False

            current_time = time.time()
            if(minuteCount != init_minute and is_behavior_danger):
                init_minute = minuteCount
                SocketTrigger.save_image(image, 'DISTRACTION')
            # Display the head pose direction
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y_angle * 10), int(nose_2d[1] - x_angle * 10))
            cv2.line(frame, p1, p2, (255, 0, 0), 3) 

            # Add the text on the frame
            cv2.putText(frame, head_pose_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

            # Drowsiness Detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray_frame, 0)
            face_rectangle = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

            for(x,y,w,h) in face_rectangle:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)

            for face in faces:
                shape = predictor(gray_frame, face)
                shape = face_utils.shape_to_np(shape)
        
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                mouth = shape[mStart:mEnd]
        
                leftEyeAspectRatio = eye_aspect_ratio(leftEye)
                rightEyeAspectRatio = eye_aspect_ratio(rightEye)
        
                eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2
                mouthAspectRatio = yawn_aspect_ratio(mouth)
        
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                mouthHull = cv2.convexHull(mouth)
        
                cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)
                cv2.drawContours(frame, [mouthHull], -1, (0,0,255), 1)

                cv2.putText(frame, f'{eyeAspectRatio}', (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)

                print(eyeAspectRatio)
                if eyeAspectRatio < EYE_ASPECT_RATIO_TRESHOLD:
                    COUNTER_EYE += 1
                    print(COUNTER_EYE)
                    if COUNTER_EYE >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                        IS_EYE_CLOSE = True
                        cv2.putText(frame, 'Mata Merem!', (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)
                        # alert.play()
                        play_sound = True
                else:
                    IS_EYE_CLOSE = False
                    COUNTER_EYE = 0
            
                if mouthAspectRatio > MOUTH_ASPECT_RATIO_TRESHOLD:
                    COUNTER_MOUTH += 1
                    if COUNTER_MOUTH >= MOUTH_ASPECT_RATIO_CONSEC_FRAMES:
                        IS_MOUTH_OPEN_15 = True
                        cv2.putText(frame, 'Mulut Mangap!', (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)
                else:
                    IS_MOUTH_OPEN_15 = False
                    COUNTER_MOUTH = 0
            
                if IS_MOUTH_OPEN_15 or IS_EYE_CLOSE:
                    IS_DROWSINESS = True
                else:
                    IS_DROWSINESS = False
    
                
                
                if IS_DROWSINESS:
                    cv2.putText(frame, 'Anda Ngantuk!', (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 4)
                    current_time = time.time()
                    if(minuteCount != init_minute):
                        init_minute = minuteCount
                        SocketTrigger.save_image(image, 'DROWSINESS')


    # Display the processed frame
    cv2.imshow('Head Pose Estimation and Drowsiness Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and output objects
cap.release()
cv2.destroyAllWindows()
