# 이름과 학번을 작성해주세요
__author__ = "조명재"
__id__ = "2018312990"

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')



class my_MLP():
    def __init__(self):
        """
        이 함수는 수정할 필요가 없습니다.
        """

        self.author = __author__
        self.id = __id__

        self.lr = 0.01
        self.neurons = 32
        self.N_hidden = 1
        self.batch_size = 8
        self.epochs = 20
        self.hidden_layers = []
        self.weights = []
        self.N_class = None
        

    def sigmoid(self, x):
        """
        주어진 x에 대하여 sigmoid 함수를 코딩하세요.
        단, 파이썬 패키지인 math를 불러와서 사용할 수 없습니다. numpy만을 사용하여 코딩하세요.
        """

        return 1.0 / (1 + np.exp(x)) 
        # sigmoid 함수 : 1/(1+exp(x)), 가장 간단하게 구현하였다.
        # 출력값이 float인 것을 나타내기 위하여 1.0으로 표기하였다. 


    def softmax(self, x):
        """
        주어진 x에 대하여 softmax 함수를 코딩하세요.
        단, 파이썬 패키지인 math를 불러와서 사용할 수 없습니다. numpy만을 사용하여 코딩하세요.
        """
        
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        # row별 합을 계산하기 위해 np.sum을 사용했으며, axis=1 으로 설정하였다.
        # keepdims=True를 통해 연산 결과가 (n, 1)형태의 array로 나오도록 설정하였다.
        # 이를 통해 간단한 나눗셈 기호로 연산이 가능케 하였으며 input과 output의 shape가 동일하다.

    def prime_sigmoid(self, x):
        """
        이 함수는 수정할 필요가 없습니다.

        해당 함수는 주어진 x에 대하여 sigmoid 함수를 미분한 식을 연산하는 함수입니다.
        """
        
        return x * (1 - x)


    def feed_forward(self, X_train):
        """
        주어진 X_train에 대하여 순전파 작업을 진행하세요.
        현재 hidden layer 정보를 weight와 곱한 값에 대해서 sigmoid 연산을 취한 값이 다음 hidden layer 정보가 됩니다.
        이를 진행하는 for문 안의 내용을 작성해주세요.

        """

        hidden_layer = X_train # 첫번째 hidden layer 정보는 X_train 값
        self.hidden_layers[0] = hidden_layer # X_train 값을 첫번째 hidden layer로 저장

        for i, weights in enumerate(self.weights):
            self.hidden_layers[i+1] = (self.sigmoid(np.dot(self.hidden_layers[i], weights)))
            # 1. i번째 hidden layer와 i번째 weight값을 내적
            # 2. activation function인 self.sigmoid에 내적한 값을 전달
            # 3. output을 hidden_layers 리스트에 순서대로 배정

        output = self.softmax(self.hidden_layers[-1]) # 마지막 layer에 대해 softmax를 취함으로써 최종 값 도출
        return output

        
    def back_propagation(self, output, y_onehot):
        """
        주어진 순전파 결과인 output과 정답 정보인 y_onehot에 대해 역전파 작업을 진행하세요.
        
        이를 진행하는 for문 안의 내용을 작성해주세요.

        """
        
        delta_t = (output - y_onehot) * self.prime_sigmoid(self.hidden_layers[-1])

        for i in range(1, len(self.weights)+1):
            self.weights[-i] -= np.dot(self.hidden_layers[-i-1].T, delta_t) * self.lr
            delta_t = np.dot(self.weights[-i], delta_t.T).T * self.prime_sigmoid(self.hidden_layers[-i-1])
        # 2번째 hidden layer : (8, 32) 2번째 delta_t : (8, 10), 2번째 weights = (32, 10)
        # 주어진 형태가 나올 수 있도록 np.dot을 통해서 행렬 곱 계산, Gradient 연산
        # 2번째 weight : (32, 10), 2번째 delta_t : (8, 10), (8, 32) 형태가 나올 수 있도록 행렬 곱 계산.
        # 같은 형태의 값을 가지고 있는 prime_sigmoid 함수의 출력값을 곱해줌.
        # 계속해서 진행하여 weights와 delta_t의 업데이트 진행

    def fit(self, X_train, y_train):
        """
        이 함수는 수정할 필요가 없습니다.

        해당 함수는 주어진 훈련 데이터 X_train와 y_train을 통해 MLP 모델을 훈련시키는 함수입니다.
        """

        self.N_class = len(np.unique(y_train)) # 분류할 클래스의 개수 설정
        y_onehot = np.eye(self.N_class)[y_train] # y_train 정보를 원핫인코딩 변환

        # 훈련 레이어의 크기 정보를 담은 리스트 : [input layer size, hidden layer size(neuron size * layer 수), output layer size]
        total_layer_size = np.array([X_train.shape[1]] + [self.neurons]*self.N_hidden + [y_onehot.shape[1]])
        self.hidden_layers = [np.empty((self.batch_size, layer)) for layer in total_layer_size] # 훈련 레이어 정보를 담을 리스트
        
        # 초기 랜덤 가중치 할당
        self.weights = list()
        for i in range(total_layer_size.shape[0]-1): # 전체 레이어 사이사이 가중치 생성
            self.weights.append(np.random.uniform(-1,1,size=[total_layer_size[i], total_layer_size[i+1]]))
        self.weights = np.array(self.weights)

        # epoch 만큼 훈련 반복
        for epoch in range(self.epochs):
            shuffle = np.random.permutation(X_train.shape[0]) # 랜덤한 훈련 데이터 index 생성
            X_batches = np.array_split(X_train[shuffle], X_train.shape[0]/self.batch_size) # X batch 생성
            y_batches = np.array_split(y_onehot[shuffle], X_train.shape[0]/self.batch_size) # y batch 생성
            
            for x_batch, y_batch in zip(X_batches, y_batches):
                output = self.feed_forward(x_batch) # 순전파 단계
                self.back_propagation(output, y_batch) # 역전파 단계 (교안의 cost function 계산과 함께 진행)


    def predict(self, X_test):
        """
        이 함수는 수정할 필요가 없습니다.

        해당 함수는 훈련된 가중치를 기반으로 평가 데이터 X_test의 예측값 y_pred를 연산하는 함수입니다.
        """

        # 평가 레이어의 크기 정보를 담은 리스트
        test_layer_size = np.array([X_test.shape[1]] + [self.neurons]*self.N_hidden + [self.N_class])
        self.hidden_layers = [np.empty((X_test.shape[0], layer)) for layer in test_layer_size] # 평가 레이어 정보를 담을 리스트

        output = self.feed_forward(X_test) # 순전파
        y_pred = output.argmax(axis=1) # 최종 예측값

        return y_pred

    

















