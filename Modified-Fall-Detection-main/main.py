import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
from tensorflow.keras.layers import LSTM

labels = np.array(['FALL','LYING','SIT','STAND','MOVE'])

n_time_steps = 25
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

def custom_lstm(*args, **kwargs):
    kwargs.pop('time_major', None)
    return LSTM(*args, **kwargs)

model = tf.keras.models.load_model('bro.h5', custom_objects={'LSTM': custom_lstm})

cap = cv2.VideoCapture(1) 

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img, label):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        # print(id, lm)
        if not label == "FALL":
            
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        else:
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
    return img


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img


def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    results = model.predict(lm_list)
    print(results)
    if results[0][0] >= 0.5: 
        label = labels[0]
        print(label)
    elif results[0][1] >= 0.5:
        label = labels[1]  
        print(label)

    elif results[0][2] >= 0.5:
        label = labels[2]
        print(label)
 
    elif results[0][3] >= 0.5:
        label = labels[3]
        print(label)
        
    elif results[0][4] >= 0.5:
        label = labels[4]
        print(label)
    else:
        label = "NONE DETECTION"

    return label

label = 'Starting...'

lm_list=[]
while True:
    
    success, img = cap.read()
    if not success:
        print("Error: Không thể lấy khung hình từ camera/video.")
        break
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    
    if results.pose_landmarks:
        c_lm = make_landmark_timestep(results)
        img = draw_landmark_on_image(mpDraw, results, img, label)
        img = draw_class_on_image(label, img)
        lm_list.append(c_lm)
        if len(lm_list) == n_time_steps:
            label = detect(model, lm_list)
            lm_list = []
    
    
            
            # img = draw_landmark_on_image(mpDraw, results, img)

        # img = draw_class_on_image(label, img)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()