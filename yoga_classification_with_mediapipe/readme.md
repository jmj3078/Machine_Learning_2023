catboost_info
model
model_reduced
resnet50
VGG16
xgboost
yogaconv2d
-------------------- 모델의 가중치가 들어있는 폴더입니다.
metrics 
-------------------- 딥러닝 모델의 학습 과정 metrics 그래프가 저장되어 있는 폴더입니다.

new_test
new_train
-------------------- Mediapipe를 사용해 생성한 새로운 이미지 데이터셋이 들어있는 폴더입니다.

catboost.ipynb
XGboost.ipynb 
-------------------- CatBoost와 XGboost를 실행한 노트북입니다.
각각 데이터 호출, parameter tuning, 테스트 데이터 호출, 예측결과 반환까지 구현되어 있습니다.

test_landmark.csv
train_landmark.csv
-------------------- 추출한 landmark의 좌표값이 csv파일 형태로 저장되어 있습니다. 총 32종의 landmark에 대하여 X, Y, Z, V 4개의 값이 저장되어 있습니다. 

poseLandmark_test.py
poseLandmark_train.py
-------------------- landmark 추출 코드입니다.
아래의 Github에서 참고하여 사용했습니다. 
https://github.com/naseemap47/CustomPose-Classification-Mediapipe

resnet50.py
VGG16.py
yogaconvo2d_modeling.py
-------------------- 새롭게 생성한 이미지를 가지고 학습을 진행한 코드입니다.

NN_modeling.py
-------------------- 간단한 Neural Network모델을 만들어 예측을 진행한 모델입니다. 가장 좋은 성능을 보인 모델입니다.

run.py
run.sh
-------------------- 테스트 데이터셋의 landmark를 추출하기 위해 사용했던 코드입니다. 불안정한 landmark 추출 과정을 최소화 하기 위해 각 이미지에 따로따로 파이썬 코드를 적용하는 방식을 사용했습니다. 한번에 반복문으로 사용하면 많은 양의 이미지가 손실되는 경향을 확인하였습니다.


