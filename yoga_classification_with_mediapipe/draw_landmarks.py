import pandas as pd 
import mediapipe as mp
import numpy as np
from PIL import Image
from mediapipe.framework.formats import landmark_pb2
import cv2
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

data = pd.read_csv('train_landmark.csv')
landmark_list = data.values[:,:-1]
label_list = data.values[:,-1]

i = 0
for landmark, label in zip(landmark_list, label_list):
    image = np.zeros((75, 75, 3), dtype=np.uint8)
    row = landmark
    # landmark중 X, Y에만 해당하는 값 호출
    landmarks = [(float(row[i]), float(row[i+1])) for i in range(0, len(row), 4)]
    connections = mp_pose.POSE_CONNECTIONS # mediapipe에서 제공하는 POSE_CONNECTION을 그대로 사용하면 됨

    for connection in connections:
        x0, y0 = landmarks[connection[0]]
        x1, y1 = landmarks[connection[1]]
        x0 = int((x0 + 1) * 37.5)  # 75X75 크기에 맞춰 rescaling
        y0 = int((y0 + 1) * 37.5)  # 75X75 크기에 맞춰 rescaling
        x1 = int((x1 + 1) * 37.5)  # 75X75 크기에 맞춰 rescaling
        y1 = int((y1 + 1) * 37.5)  # 75X75 크기에 맞춰 rescaling
        cv2.line(image, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=1)
                
    for x, y in landmarks:
        x = int((x + 1) * 37.5)  # 75X75 크기에 맞춰 rescaling
        y = int((y + 1) * 37.5)  # 75X75 크기에 맞춰 rescaling
        cv2.circle(image, (x, y), radius=1, color=(0, 0, 255), thickness=-1)
  
    save_path = f"./new_train/{label}/{i}.jpg"
    cv2.imwrite(save_path,image)
    i += 1 
print("drawing landmark of train image finished")


test_landmarks = pd.read_csv("test_landmark.csv")
test = pd.read_csv("test_landmark.csv")
test_X = test.iloc[:,:-1]
nan_rows = test_X.isnull().any(axis=1)
# 일부 landmark를 뽑아올 수 없었던 데포르메 이미지에 대한 무작위 imputation
nan_idx = nan_rows[nan_rows].index
for idx in nan_idx:
    test_X.loc[idx] = test.loc[np.random.choice(test.index)]

i = 0
for landmark in test_X.values:
    image = np.zeros((75, 75, 3), dtype=np.uint8)
    row = landmark
    landmarks = [(float(row[i]), float(row[i+1])) for i in range(0, len(row), 4)]
    connections = mp_pose.POSE_CONNECTIONS
            
    for connection in connections:
        x0, y0 = landmarks[connection[0]]
        x1, y1 = landmarks[connection[1]]
        x0 = int((x0 + 1) * 37.5)  # 75X75 크기에 맞춰 rescaling
        y0 = int((y0 + 1) * 37.5)  # 75X75 크기에 맞춰 rescaling
        x1 = int((x1 + 1) * 37.5)  # 75X75 크기에 맞춰 rescaling
        y1 = int((y1 + 1) * 37.5)  # 75X75 크기에 맞춰 rescaling
        cv2.line(image, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=1)
    
    for x, y in landmarks:
        x = int((x + 1) * 37.5)  # 75X75 크기에 맞춰 rescaling
        y = int((y + 1) * 37.5)  # 75X75 크기에 맞춰 rescaling
        cv2.circle(image, (x, y), radius=1, color=(0, 0, 255), thickness=-1)

    save_path = f"./new_test/{i}.jpg"
    cv2.imwrite(save_path,image)
    i += 1 
print("drawing landmark of test image finished")