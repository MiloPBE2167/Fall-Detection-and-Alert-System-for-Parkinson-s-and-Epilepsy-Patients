import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM
import streamlit as st
import time
import pygame
import time
import threading

labels = np.array(['FALL', 'LYING', 'SIT', 'STAND', 'MOVE'])

n_time_steps = 25
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
start_time = None
def custom_lstm(*args, **kwargs):
    kwargs.pop('time_major', None)
    return LSTM(*args, **kwargs)

model = tf.keras.models.load_model('bro.keras', custom_objects={'LSTM': custom_lstm})

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img, label):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        if label != "FALL":
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        else:
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
    return img

def draw_bounding_box_around_people(results, img, label, alert=False):
    h, w, c = img.shape

    x_min, y_min = w, h
    x_max, y_max = 0, 0
    
    for id, lm in enumerate(results.pose_landmarks.landmark):
        cx, cy = int(lm.x * w), int(lm.y * h)
        if id == 0:
            pose_nose_y = cy
        if id == 12:
            pose_shoulder_y = cy
        if cx > 0 and cx < w:
            x_min = min(x_min, cx)
            x_max = max(x_max, cx)
        
        if cy > 0 and cy < h:
            y_max = max(y_max, cy)
            y_min = min(y_min, cy)
         
    if pose_shoulder_y-((pose_shoulder_y - pose_nose_y)*2) > 0:
        y_min = min(pose_shoulder_y-((pose_shoulder_y - pose_nose_y)*2),y_min)
    else:
        y_min = 0
 
    if x_max > x_min and y_max > y_min:
        color = (0, 255, 0) if label != "FALL" else (0, 0, 255)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Add a flashing red circle around the bounding box in case of fall detection
        if label == "FALL" and alert:
            cv2.circle(img, (int((x_min + x_max) / 2), int((y_min + y_max) / 2)), 100, (0, 0, 255), 10)

def draw_class_on_image(label, img, alert=False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    h,w,c = img.shape
    bottomLeftCornerOfText = (20, h-10)
    fontScale = 1
    fontColor_normal = (0, 255, 0)
    fontColor_fall = (0, 0, 255)
    thickness = 4
    lineType = 2
    if label == "FALL" and alert:
        fontScale = 1.5
       
        cv2.putText(img, "WARNING: FALL DETECTED!", (10, 60), font, fontScale, (0, 0, 255), thickness, lineType)
        
    (text_width, text_height), baseline = cv2.getTextSize(label, font, fontScale, thickness)
    cv2.rectangle(img,(0,h-20-text_height),(20+text_width+20,h),(255,255,255),-1)
    cv2.putText(img, label, bottomLeftCornerOfText, font, fontScale, fontColor_fall if alert else fontColor_normal, thickness, lineType)
    return img

def detect(model, lm_list):
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)
    if results[0][0] >= 0.5: 
        label = labels[0]
    elif results[0][1] >= 0.5:
        label = labels[1]  
    elif results[0][2] >= 0.5:
        label = labels[2]
    elif results[0][3] >= 0.5:
        label = labels[3]
    elif results[0][4] >= 0.5:
        label = labels[4]
    else:
        label = "NONE DETECTION"
    return label


def Phat_am_thanh_canh_bao(label):
    

    if label == "FALL":
        path_to_sound = "caution_music.mp3"
        so_phut_canh_bao = 60
        pygame.mixer.init()
        sound = pygame.mixer.Sound(path_to_sound)
        
        def play_sound():
            so_lan_lap = int(so_phut_canh_bao / sound.get_length())
            for _ in range(so_lan_lap):
                sound.play()
                while pygame.mixer.get_busy():
                    time.sleep(0.1)
        
 
        sound_thread = threading.Thread(target=play_sound)
        sound_thread.start()
 
     





def main():
    st.title("Pose Detection and Classification")
    
    run_type = st.sidebar.selectbox("Select input type", ("Camera", "Video File"))
    
    if run_type == "Camera":
        cap = cv2.VideoCapture(1)
    else:
        video_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        if video_file is not None:
            with open("temp_video.mp4", "wb") as f:
                f.write(video_file.read())
            cap = cv2.VideoCapture("temp_video.mp4")
        else:
            st.write("Please upload a video file.")
            return
    
    stframe = st.empty()
    label = 'Starting...'
    lm_list = []
    alert = False

    while cap.isOpened():
        start_time = time.time()
        success, img = cap.read()
        if not success:
            break
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        
        if results.pose_landmarks:
            c_lm = make_landmark_timestep(results)
            img = draw_landmark_on_image(mpDraw, results, img, label)
            img = draw_class_on_image(label, img, alert)
            draw_bounding_box_around_people(results, img, label, alert)
            lm_list.append(c_lm)
            if len(lm_list) == n_time_steps:
                label = detect(model, lm_list)
                lm_list = []
                alert = (label == "FALL")  # Trigger alert if fall is detected
                Phat_am_thanh_canh_bao(label)
        else:
            if start_time == None:
                start_time = time.time()
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.5:
                lm_list = []
                label = "NONE DETECTION"
        stframe.image(img, channels="BGR")
        
        if alert:
            time.sleep(0.1)  # Flashing effect for fall alert
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()

if __name__ == '__main__':
    main()
