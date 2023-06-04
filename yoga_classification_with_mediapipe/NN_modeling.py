
import keras
import pandas as pd
import tensorflow as tf
from keras import layers, Sequential
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os
import numpy as np
from sklearn.utils import class_weight


df = pd.read_csv('./train_landmark.csv')
class_list = df['Pose_Class'].unique()
class_list = sorted(class_list)
class_number = len(class_list)

x = df.copy()
y = x.pop('Pose_Class')

y, _ = y.factorize()
x = x.astype('float64')

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=42)
 
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(y_train),
                                                 y=y_train)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

class_weights = dict(enumerate(class_weights))

tf.random.set_seed(42)

# 간단한 NN모델 제작
model = Sequential([
    layers.Dense(2048, activation='tanh', input_shape=[x_train.shape[1]]),
    layers.Dropout(0.2),
    layers.Dense(1024, activation='tanh'),
    layers.Dropout(0.2),
    layers.Dense(512, activation='tanh'),
    layers.Dropout(0.2),
    layers.Dense(256, activation='tanh'),
    layers.Dropout(0.5),
    layers.Dense(class_number, activation="softmax")
])

# Model Summary 출력
print('Model Summary: ', model.summary())
# 모델 compile, checkpoint와 earlystopping 설정
# M1 맥 환경이기 때문에 lagacy.Adam 사용
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 가장 높은 Val_accuracy를 가지는 곳에서 Checkpoint형성 후 저장
# Checkpoint 저장경로 설정, Early stopping 설정
checkpoint_path = './model'
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                             monitor='val_accuracy',
                                             verbose=3,
                                             save_best_only=True,
                                             mode='max')
earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                              patience=100)

# 모델 Training 시작, checkpoint와 earlystopping옵션 적용
# Hyperparmaeter적용 
history = model.fit(x_train, y_train,
                    epochs=1000,
                    batch_size=512,
                    validation_data=(x_test, y_test),
                    class_weight=class_weights,
                    callbacks=[checkpoint, earlystopping])

# 모델 Training 완료 메세지
print('Model Training Completed')
print(f'Model Successfully Saved in {checkpoint_path}')

# Test landmark를 불러와 predict.csv 결과 형성
test = pd.read_csv("test_landmark.csv")
test_X = test.iloc[:,:-1]
nan_rows = test_X.isnull().any(axis=1)
# 일부 landmark를 뽑아올 수 없었던 데포르메 이미지에 대한 무작위 imputation
nan_idx = nan_rows[nan_rows].index
for idx in nan_idx:
    test_X.loc[idx] = test.loc[np.random.choice(test.index)]
# 예측결과 추출
predict_proba = model.predict(test_X)
predict = predict_proba.argmax(axis=1)
Id = pd.Series([f"{i}.jpg" for i in range(0,389)])

# 데이터프레임으로 결과 변환, 저장
result = pd.DataFrame({'Id':Id, 'Category':predict})
result.to_csv("predict_NN.csv", index=False)

# History plotting. 그림으로 학습 결과 확인
# history에서 데이터 불러오기
metric_loss = history.history['loss']
metric_val_loss = history.history['val_loss']
metric_accuracy = history.history['accuracy']
metric_val_accuracy = history.history['val_accuracy']

# 최고 val_accuracy출력
print(max(metric_val_accuracy))
# X축 범위(epoch수) 설정
epochs = range(len(metric_loss))

# 그래프 plotting
plt.plot(epochs, metric_loss, 'blue', label=metric_loss)
plt.plot(epochs, metric_val_loss, 'red', label=metric_val_loss)
plt.plot(epochs, metric_accuracy, 'blue', label=metric_accuracy)
plt.plot(epochs, metric_val_accuracy, 'green', label=metric_val_accuracy)

# title추가
plt.title(str('Model Metrics'))

# legend추가
plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'])

# 이미 존재하는 이미지가 있으면 삭제(버그 발생)
plt.savefig('NN_metrics.png', bbox_inches='tight')
print('Successfully Saved metrics.png')
