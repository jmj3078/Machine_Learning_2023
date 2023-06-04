import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
import argparse
import glob as glob
import csv

# 리눅스 커널에서 argparse를 사용해 이미지의 path 받음
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--source", type=str, required=True,
                help="path to test image")
args = vars(ap.parse_args())
path = args["source"]

# 각 landmark의 이름 설정, 설정하지 않을 시 x1,y1,z1,v1...로 출력
# 데이터 해석에 큰 어려움이 발생하기 때문에 사용함.
torso_size_multiplier = 2.5
n_landmarks = 33
n_dimensions = 3
landmark_names = [
    'nose',
    'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear',
    'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_pinky_1', 'right_pinky_1',
    'left_index_1', 'right_index_1',
    'left_thumb_2', 'right_thumb_2',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index',
]
# 클래스 이름 설정
class_names = [
    0, 1, 2, 3, 4, 5
]

# mediapipe 모델 호출
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 저장할 csv파일의 column name 설정
col_names = []
for i in range(n_landmarks):
    name = mp_pose.PoseLandmark(i).name
    name_x = name + '_X'
    name_y = name + '_Y'
    name_z = name + '_Z'
    name_v = name + '_V'
    col_names.append(name_x)
    col_names.append(name_y)
    col_names.append(name_z)
    col_names.append(name_v)

# 저장할 csv파일 오픈, csv.writer를 통해서 한줄씩 입력
f = open("/Users/mjcho/Downloads/project3/Mediapipe-yoga/test_landmark.csv","a")
writer = csv.writer(f)

# 이미지 호출, RGB변환(cv2)
img = cv2.imread(path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = pose.process(img_rgb)

# pose.landmark가 호출 되었을 경우 진행
# hip에 대한 landmark를 중심으로 설정하고, normalizaed된 좌표값을 설정하는 과정
if result.pose_landmarks:
    lm_list = []
    for landmarks in result.pose_landmarks.landmark:
        max_distance = 0
        lm_list.append(landmarks)
    center_x = (lm_list[landmark_names.index('right_hip')].x +                    
                lm_list[landmark_names.index('left_hip')].x)*0.5
    center_y = (lm_list[landmark_names.index('right_hip')].y +
                lm_list[landmark_names.index('left_hip')].y)*0.5

    shoulders_x = (lm_list[landmark_names.index('right_shoulder')].x +
                lm_list[landmark_names.index('left_shoulder')].x)*0.5
    shoulders_y = (lm_list[landmark_names.index('right_shoulder')].y +
                lm_list[landmark_names.index('left_shoulder')].y)*0.5

    for lm in lm_list:
        distance = math.sqrt((lm.x - center_x)**2 + (lm.y - center_y)**2)
        if(distance > max_distance):
            max_distance = distance
    torso_size = math.sqrt((shoulders_x - center_x) **
                        2 + (shoulders_y - center_y)**2)
    max_distance = max(torso_size*torso_size_multiplier, max_distance)
    # landmark 호출 결과 저장
    pre_lm = list(np.array([[(landmark.x-center_x)/max_distance, (landmark.y-center_y)/max_distance,
                                landmark.z/max_distance, landmark.visibility] for landmark in lm_list]).flatten())
    pre_lm.append(path)
    # write.row를 사용하여 한줄 씩 저장
    writer.writerow(pre_lm)

else : 
    # 랜드마크 추출이 안되는 이미지를 구별하기 위해 추가한 코드
    # 결측값으로 처리해서 csv에 저장하고 그 이름까지 전달한다
    print(f"No pose landmarks found in image: {path}")
    nan = [ 'NaN' for i in range(len(col_names))]
    nan.append(path)
    writer.writerow(nan)

f.close()