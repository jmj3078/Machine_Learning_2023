import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
import numpy as np
import cv2

input_directory = "new_train"
IMG_SIZE=(75, 75)
# 데이터셋 호출
train_dataset = image_dataset_from_directory(input_directory,
                                             shuffle=True,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='training',
                                             seed=42)
validation_dataset = image_dataset_from_directory(input_directory,
                                             shuffle=True,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='validation',
                                             seed=42)

class_names = train_dataset.class_names
print(class_names)

# Transfer learning, VGG16사용
base_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(75, 75, 3))
base_model.trainable=True

model = keras.models.Sequential([
    keras.layers.Input(IMG_SIZE+(3,)),
    RandomFlip('horizontal'),
    RandomRotation(0.2),
    base_model,
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(6, activation='softmax')  
])
model.build()
model.summary()

# 모델 compile, checkpoint와 earlystopping 설정
# M1 맥 환경이기 때문에 lagacy.Adam 사용
checkpoint_path='./VGG16'
model.compile(
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
)
# checkpoint, Earlystopping 설정, 가장 좋은 validation accuracy 결과가 나왔을 때 저장되도록 설정
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                             monitor='val_accuracy',
                                             verbose=3,
                                             save_best_only=True,
                                             mode='max')
earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                              patience=30)

# 모델 Training 시작, checkpoint와 earlystopping옵션 적용
# Hyperparmaeter적용 
history = model.fit(train_dataset,
                    epochs=200,
                    batch_size=64,
                    validation_data=validation_dataset,
                    callbacks=[checkpoint, earlystopping])
print('Model Training Completed')
print(f'Model Successfully Saved in {checkpoint_path}')

# History plotting. 그림으로 학습 결과 확인
# history에서 데이터 불러오기
metric_loss = history.history['loss']
metric_val_loss = history.history['val_loss']
metric_accuracy = history.history['accuracy']
metric_val_accuracy = history.history['val_accuracy']

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
plt.savefig('VGG16_metrics.png', bbox_inches='tight')
print('Successfully Saved metrics.png')

model = keras.models.load_model(checkpoint_path)

# Test이미지 불러오고, test 시작, 예측값 생성
test_path = [f'./new_test/{i}.jpg' for i in range(0, 389)]
predict = []
for path in test_path:
    img = cv2.imread(path)
    img = np.expand_dims(img, axis=0)
    proba = model.predict(img)
    pred = proba.argmax(axis=1)
    predict.append(pred)

predict = np.concatenate(predict)
Id = pd.Series([f"{i}.jpg" for i in range(0,389)])

# 데이터프레임으로 결과 변환, 저장
result = pd.DataFrame({'Id':Id, 'Category':predict})
result.to_csv("predict_VGG16.csv", index=False)


