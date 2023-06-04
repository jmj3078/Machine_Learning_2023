# 이름과 학번을 작성해주세요
__author__ = "조명재"
__id__ = "2018312990"

# 작성되어있는 라이브러리 이외의 어떠한 라이브러리도 호출할 수 없습니다.
import numpy as np

########  Notice! 평가 기준  #########
#  activation 함수 : 0.5점          #
#  fwpass 함수 : 0.5점              #
#  bwpass 함수 : 0.5점              #
#  initialize_w 함수 : 0.5점        #
#  fit 함수 : 0.5점                 #
#  predict 함수 : 0.5점             #
#  feature_importance 함수 : 1점    #

#  time 0.07초 이하 : 0.5점         #
#  accuracy 0.75 이상 : 0.5점       #
#  코드 내의 주석                    #
############  총 5점 부여  ###########


class LogisticRegression:
    def __init__(self, max_iter=500, penalty="l2", initialize = "one", random_seed = 1213):
        """
        이 함수는 수정할 필요가 없습니다.
        """
        self.author = __author__
        self.id = __id__
        
        self.max_iter = max_iter
        self.penalty = penalty
        self.initialize = initialize
        self.random_seed = random_seed
        self.lr = 0.1
        self.lamb = 0.01
        np.random.seed(self.random_seed)

        if self.penalty not in ["l1", "l2"]:
            raise ValueError("Penalty must be l1 or l2")

        if self.initialize not in ["one", "LeCun", "random"]:
            raise ValueError("Only [LeCun, One, random] Initialization supported")

            
    def activation(self, z):
        """
        주어진 z에 대하여 sigmoid 활성화 함수를 코딩하세요.
        """
        a = 1.0/(1 + np.exp(-z)) # Code Here!
        # 가장 간단하게 시그모이드 함수를 구현하였다.
        return a


    def fwpass(self, x):
        """
        x가 주어졌을 때, 가중치 w와 bias인 b를 적절히 x와 내적하여
        아래의 식을 계산하세요.

        z = w1*x1 + w2*x2 + ... wk*xk + b
        """
        # Code Here! 이 부분에 w1*x1 + w2*x2 + ... wk*xk + b 의 값을 가지도록 계산하세요. (넘파이 행렬 활용 추천) 
        z = np.dot(x, self.w) + self.b 
        # np.dot 사용하여 가중치와 x를 내적하고 bias를 더해줌
        # linear regression의 형태와 동일하다.
        z = self.activation(z)
        return z


    def bwpass(self, x, err):
        """
        x와 오차값인 err가 들어왔을 때, w와 b에 대한 기울기인 w_grad와 b_grad를 구해서 반환하세요.
        l1, l2을 기반으로한 미분은 https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261
        이 문서를 확인하세요.

        w_grad는 (num_data, num_features)
        b_grad는 (num_data, )

        의 데이터 shape을 가지는 numpy array 입니다.
        어떠한 방법을 활용해서 w_grad를 계산해도 괜찮으며, 이 부분에서 속도의 차이가 발생합니다.

        self.lamb을 통해 lambda를 활용하세요.
        """
        n = x.shape[1]
        if self.penalty == "l1":
            w_grad = np.dot(err, x) + self.lamb * abs(self.w)
            # logistic regresssion (L1) 에서의 cost function :
            # J(w) = - [y * log(sigmoid(X·w)) + (1 - y) * log(1 - sigmoid(Xw))] + λ * ||w||1
            # ∇J(w) = X.T * (sigmoid(X·w) - y) + λ * |w| 이다.
            # 모두 obs의 크기인 n으로 나눠줘야 하나 이를 갱신 과정에 적용했다.
            # sigmaoid(X·w) - y = err 이므로, 간단하게 위와 같이 낱타낼 수 있다.

        elif self.penalty == "l2":
            w_grad = np.dot(err, x) + self.lamb * (np.power(self.w, 2))
            # logistic regresssion (L2) 에서의 cost function :
            # J(w) = - [y * log(sigmoid(X·w)) + (1 - y) * log(1 - sigmoid(Xw))] + λ * ||w||2
            # ∇J(w) = X.T * (sigmoid(X·w) - y) + λ * w^2 이다.
            # 모두 obs의 크기인 n으로 나눠줘야 하나 이를 갱신 과정에 적용했다.
            # sigmaoid(X·w) - y = err 이므로, 간단하게 위와 같이 나타낼 수 있다.


        b_grad = np.mean(err)
        # grad_b = 1/n * sum(sigmoid(X·w) - y).
        # sigmoid(X·w) - y = err 이므로, 위와 같이 평균내어 간단하게 표현할 수 있다.
        # 참고 링크 : https://medium.com/analytics-vidhya/logistic-regression-with-gradient-descent-explained-machine-learning-a9a12b38d710

        return w_grad, b_grad

    
    def initialize_w(self, x):
        """
        L8(이번주차 강의)-NN-GD2와 https://reniew.github.io/13/ 의 LeCun 초기화 수식을 참고하여
        LeCun 가중치 초기화 기법으로 초기 w를 설정할 수 있도록 코딩하세요. (힌트: np.random.uniform 활용)
        동일하게 랜덤한 값으로 w가중치를 초기화 하세요.

        단, numpy 만을 사용하여야 하며, 다른 라이브러리는 사용할 수 없습니다.
        w_library에서 one과 같은 shape이 되도록 다른 값을 설정하세요.
        """
        size = np.shape(x)[1]
        w_library = {
            "one":np.ones(size),
            "LeCun":np.random.uniform(-np.sqrt(1/size), np.sqrt(1/size), size), # Code Here! LeCun 식을 활용하여 w가중치를 초기화 할 수 있도록 수식을 작성하세요. 
            # Lecun - uniform distribution : (1/sqrt(n)), 1/sqrt(n) 범위의 균등분포
            # Lecun - Normal distribution : np.random.normal(0, np.sqrt(1/size), size)를 적용할 수 있음
            "random":np.random.rand(size) # Code Here! 랜덤한 0~1사이의 값으로 w가중치를 초기화 할 수 있도록 수식을 작성하세요. 
            # 랜덤판 0~1 사이의 (size,) 크기의 array 반환
        }

        return w_library[self.initialize]

    def fit(self, x, y):
        """
        실제로 가중치를 초기화 하고, 반복을 진행하며 w, b를 미분하여 계산하는 함수입니다.
        다른 함수를 통하여 계산이 수행되니 self.w, self.b 의 업데이트 과정만 코딩하세요.
        """
        self.w = self.initialize_w(x)
        self.b = 0
        for _ in range(self.max_iter):
            z = self.fwpass(x)
            err = -(y - z)
            w_grad, b_grad = self.bwpass(x, err)
            # Code Here! w를 w_grad를 활용하여 업데이트하세요. (각 gradient에 learning_rate을 곱한 후 평균을 활용하여 값을 업데이트)
            # 어떠한 방법을 사용해서 업데이트 해도 좋으며, 이 부분에서 속도의 차이가 발생합니다.
            self.w -= self.lr * w_grad / x.shape[1]
            # Gradient 계산 과정에 1/n 계산을 빼먹었기 때문에 이곳에 적용하여, gradient vector를 학습률과 곱하여 갱신한다.

            # Code Here! b를 b_grad를 활용하여 업데이트 하세요. 
            # 어떠한 방법을 사용해서 업데이트 해도 좋으며, 이 부분에서 속도의 차이가 발생합니다.
            self.b -= self.lr * b_grad 
            # bias도 마찬가지로 gradient 값을 학습률과 곱하여 갱신
        return self.w, self.b


    def predict(self, x):
        """
        test용 x가 주어졌을 때, fwpass를 통과한 값을 기반으로
        0.5초과인 경우 1, 0.5이하인 경우 0을 반환하세요.
        """
        z = self.fwpass(x)
        z[z > 0.5] = 1
        z[z <= 0.5] = 0
        # 가장 간단한 boolean indexing으로 진행하였다. 
        return z 

    
    def score(self, x, y):
        """
        이 함수는 수정할 필요가 없습니다.
        """
        return np.mean(self.predict(x) == y)
    
    
    def feature_importance(self, coef, column_to_use):
        """
        본 과제에서 사용한 feature들의 중요도를 '순서대로' 보여주세요.
        함께 제공된 ipynb 파일의 예시처럼 "순위, feature, 해당 feature의 가중치"가 모두 나타나야합니다.
        print 함수를 사용해 하나하나 출력한 것은 인정하지 않습니다.
        가중치의 순위를 활용하여 코딩하세요. (힌트: np.argsort)
        """
        rank = np.argsort(abs(coef))[::-1]
        # coefficient의 절댓값을 통해 중요도를 간단하게 비교할 수 있기 때문에
        # 절댓값을 적용한 후 np.argsort()[::-1]를 사용해 내림차순으로 정렬했다.
        # 이후 enumerate(rank)를 사용하여 주어진 feature와 coef에 순위대로 접근하여 출력하였다.
        for i, j in enumerate(rank) :
            print(f"rank : #{i+1}, feature : {column_to_use[j]}, weight : {coef[j]}")
        return 
