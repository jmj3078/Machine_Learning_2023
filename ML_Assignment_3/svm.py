

# 이름과 학번을 작성해주세요
__author__ = "조명재"
__id__ = "2018312990"

# 작성되어있는 라이브러리 이외의 어떠한 라이브러리도 호출할 수 없습니다.
import numpy as np
import matplotlib.pyplot as plt

# 아래에 코드를 작성해주세요.

########  Notice! 평가 기준  #########
#  fit 함수 : 1.5점                 #
#  predict 함수 : 1점               #
#  get_accuracy 함수 : 1점          #
#  visualization 함수 : 1.5점       #
############  총 5점 부여  ###########


class SVMClassifier:
    def __init__(self,n_iters=100, lr = 0.0001, random_seed=3, lambda_param=0.01):
        """
        이 함수는 수정하지 않아도 됩니다.
        """
        self.author = __author__
        self.id = __id__
        self.n_iters = n_iters # 몇 회 반복하여 적절한 값을 찾을지 정하는 파라미터
        self.lr = lr  # 학습률과 관련된 파라미터 
        self.lambda_param = lambda_param
        self.random_seed = random_seed
        np.random.seed(self.random_seed)


    def fit(self, x, y):
        """
        본 함수는 x, y를 활용하여 훈련하는 과정을 코딩하는 부분입니다.
        아래 reference 사이트의 gradient 계산 부분을 참고하세요.
        reference: https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
        아래 총 6개의 None 부분을 채우시면 됩니다.

        """
        n_samples, n_features = x.shape
        # hint: y값을 SVM 계산에 활용해주기 위하여 0에 해당하는 y값들을 -1로 변환
        y = np.where(y <= 0, -1, 1)
        
        # hint: w값 초기화, (n_features, )의 크기를 가지는 0과 1사이의 랜덤한 변수 어레이 (필수: 넘파이로 정의해야 함)
        np.random.seed(self.random_seed) # random seed 설정
        init_w = np.random.rand(n_features) # 0, 1 사이의 랜덤변수

        self.w = init_w
        self.b = 0 # b값 초기화

        for _ in range(self.n_iters):
            for i in range(n_samples):
                x_i = x[i]
                y_i = y[i]

                # hint: y(i) * (w · x(i) + b) >= 1 를 만족하는 경우의 의미가 담기도록 if문을 채우세요.
                condition = y_i * (np.dot(self.w, x_i) + self.b) >= 1 
                if condition:
                    # hint: w에 대하여 Gradient Loss Function 수식을 이용하여 W를 업데이트 하세요.
                    # gradient : 2*lamda*w
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
    
                else:
                    # hint: w에 대하여 Gradient Loss Function 수식을 이용하여 W를 업데이트 하세요.
                    # gradient : 2*lamda*w - x_i·y_i)
                    # bias = y_i
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_i))
                    self.b -= self.lr * y_i
                    
        return self.w, self.b 

        # visualization에 사용하기 위해 반환

    def predict(self, x):
        """
            [n_samples x features]로 구성된 x가 주어졌을 때, fit을 통해 계산된 
            self.w와 self.b를 활용하여 예측값을 계산합니다.

            @args:
                [n_samples x features]의 shape으로 구성된 x
            @returns:
                [n_samples, ]의 shape으로 구성된 예측값 array

            아래의 수식과 수도코드를 참고하여 함수를 완성하면 됩니다.
                approximation = W·X - b
                if approximation >= 0 {
                    output = 1
                }
                else{
                    output = 0
                }
        """
        output = np.dot(x, self.w) - self.b
        predicted_labels = np.sign(output) # 숫자 부호만을 인식하여 -1, 1로 반환
        y_hat = np.where(predicted_labels <= -1, 0, 1) # -1로 설정된 label을 다시 0으로 바꿔서 출력
        return y_hat

    def get_accuracy(self, y_true, y_pred):
        """
            y_true, y_pred가 들어왔을 때, 정확도를 계산하는 함수.
            sklearn의 accuracy_score 사용 불가능 / sklearn의 accuracy_score 함수를 구현한다고 생각하면 됩니다.
            넘파이만을 활용하여 정확도 계산 함수를 작성하세요.
        """
        n = y_true.size
        accuracy = (y_true == y_pred).sum() / n 
        return accuracy


    def visualization(self, X_test, y_test, coef, interrupt):
        """
            Test set에 대한 SVM Classification의 예측 결과를 시각화하는 함수.
            함께 제공된 ipynb 파일의 예시처럼 시각화를 수행하면 됩니다.
            Test set의 class 별 데이터를 색 구분을 통해 나타내며, 빨간색 선을 통해 학습시킨 모델이 예측한 hyperplane을 보여줍니다.

        """
        plt.scatter(X_test[:,0][y_test==0], X_test[:,1][y_test == 0],marker='o',color='blue', label=0)
        plt.scatter(X_test[:,0][y_test==1], X_test[:,1][y_test == 1],marker='o',color='orange', label=1)
        x_points = X_test[:,0]
        y_points = -(coef[0]/coef[1]) * x_points - (interrupt/coef[1])
        plt.plot(X_test[:,0], y_points, color='red', linestyle='dashed', linewidth=1)
        plt.legend()
        plt.title("SVM Classification")
        plt.draw() # plt.draw()가 없으면 저장이 되지 않아, 추가해두었습니다.
        plt.savefig(self.author + '_' + self.id + '_' + 'visualization.png')
