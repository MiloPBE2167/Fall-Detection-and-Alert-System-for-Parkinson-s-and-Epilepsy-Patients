import cv2
import mediapipe as mp
import pandas as pd
import os

duoicuavideo = "avi"
directory = 'train/Walking'
label = "WAlk"
number = 0
frame = 25

# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


def make_landmark_timestep(results):
    print(results.pose_landmarks.landmark)
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm
for filename in os.listdir(directory):
    if filename.endswith('.' + duoicuavideo):
        filepath = os.path.join(directory, filename)
        cap = cv2.VideoCapture(filepath)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        lm_list = []
        if frame_count > 30:
            for i in range(frame_count):
                ret, frame = cap.read()
                if ret:
                    # Nhận diện pose
                    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frameRGB)

                    if results.pose_landmarks:
                        # Ghi nhận thông số khung xương
                        lm = make_landmark_timestep(results)
                        lm_list.append(lm)
                        
                    cv2.imshow("image", frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
                if len(lm_list) == 25:
                    df  = pd.DataFrame(lm_list)
                    df.to_csv(label + str(number) + ".txt")
                    
            number +=1 
            cap.release()
            cv2.destroyAllWindows()