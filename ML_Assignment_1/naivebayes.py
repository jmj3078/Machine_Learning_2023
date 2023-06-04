# 이름과 학번을 작성해주세요
__author__ = "조명재"
__id__ = "2018312990"

# 넘파이 이외의 어떠한 라이브러리도 호출할 수 없습니다.
import numpy as np


# 아래에 코드를 작성해주세요.
class NaiveBayesClassifier:
    def __init__(self, smoothing=1):
        """
        이 함수는 수정하지 않아도 됩니다.
        """
        self.author = __author__
        self.id = __id__
        self.smoothing=smoothing
        self.epsilon = 1e-10

    def fit(self, x, y):

        """
        이 함수는 수정하지 않아도 됩니다.
        
        실제적인 훈련(확률 계산)이 이뤄지는 부분입니다.
        self.data에 [도큐먼트수 x 단어피쳐수(500)]인 넘파이 행렬이
        self.labels에 각 도큐먼트에 대한 라벨인 [도큐먼트수, ]인 넘파이 행렬이 저장됩니다.

        본 함수가 호출된 이후, self.label_index 변수에 아래와 같은 딕셔너리형태가 저장됩니다.
        self.label_index = {0: array([   1,    4,    5, ..., 3462, 3463, 3464]), 1: array([   0,    2,    3, ..., 3447, 3449, 3453])}
        0번 라벨의 도큐먼트 id가 넘파이 어레이 형태로, 1번 라벨의 도큐먼트 id가 넘파이 어레이 형태로 위와 같이 정리됩니다.

        이후, label_index, prior, likelihood를 순차적으로 계산하여 줍니다.
        아래에서 호출되는 self.get_... 함수들이 직접 작성하실 계산 함수입니다.
        이 함수는 수정하지 않아도 됩니다.
        """

        self.data = x
        self.labels = y

        self.label_index = dict()
        self.label_name = set(self.labels)
        for lab in self.label_name:
            self.label_index[lab] = []

        for index, label in enumerate(self.labels):
            self.label_index[label].append(index)
            
        for lab in self.label_name:
            self.label_index[lab] = np.array(self.label_index[lab])

        self.get_prior()
        self.get_likelihood()
        

    def get_prior(self):
        """
        prior를 계산하는 함수입니다.
        본 함수가 처리된 이후, self.prior 변수에 라벨이 key, 라벨에 대한 prior가 value로 들어가도록 하세요.
        self.prior = {0: 0번 라벨 prior[실수값], 1: 1번 라벨 prior[실수값]}

        단 채점시, 라벨이 2개 이상일 수도 있으므로 라벨 2개인 경우에서만 잘 작동할 경우 점수가 부여되지 않습니다.
        """
        self.prior = dict()

        for lab in self.label_name:
            self.prior[lab] = (self.label_index[lab].size)/(self.data.shape[0])
            # 사전확률 : (특정 label에 속한)데이터 수/전체 데이터 수

        return self.prior


    def get_likelihood(self):

        """
        likelihood를 계산하는 함수입니다. 
        본 함수가 처리된 이후, self.likelihood에 라벨이 key, 라벨에 대한 단어별 likelihood를 계산하여 value로 넣어주세요.

        """
        self.likelihood = {}
        for lab in self.label_name:
            self.likelihood[lab] = (self.data[self.label_index[lab]].sum(axis=0) + 1) / (self.data.sum(axis=0) + 1) 
            # laplace smoothing, 각 클래스에서 단어가 한번씩은 등장했다고 가정함
            # 분모에도 1을 더해주었다.
        
        return self.likelihood

    def get_posterior(self, x):

        """
        self.likelihood와 self.prior를 활용하여 posterior를 계산하는 함수입니다.
        0, 1 라벨에 대한 posterior를 계산하세요.

        Overflow를 막기위해 log와 exp를 활용합니다. 아래의 식을 고려해서 posterior를 계산하세요.
        posterior 
        = prior * likelihood 
        = exp(log(prior * likelihood))  refer. log(ab) = log(a) + log(b)
        = exp(log(prior) + log(likelihood))

        nan을 막기 위해 possibility 계산시에 분모에 self.epsilon을 더해주세요.

        """
        temp_matrix = np.empty((x.shape[0],0))

        for lab in self.label_name:
            log_likelihood = np.log(np.array(self.likelihood[lab]))
            log_prior = np.log(np.array(self.prior[lab]))
            temp_matrix = np.column_stack((temp_matrix, np.exp((x * log_likelihood).sum(axis=1) + log_prior)))

        # n * p 크기의 test input에 1 * p 크기의 likelihood vector를 log를 씌운 상태로 * 연산자로 곱샘 연산을 진행한다.
        # 이는 각 feature의 등장 횟수 만큼 likelihood를 곱한 값과 같다. 사전확률도 로그를 씌운 뒤 더한다.
        # 다시 np.exp을 통해서 로그를 벗겨낸 뒤, n * (label의 순서) 구조의 확률 matrix를 형성하기 위해 column_stack을 사용하였다.
        # 최종적으로 predict 함수에서 argmax(,axis=1)에 의해 확률이 가장 높은 label의 index가 나오게 된다.

        self.posterior = temp_matrix/(temp_matrix.sum(axis=1)[:,None] + self.epsilon)
        # possibility 계산, epsilon 추가


        return self.posterior


    def predict(self, x):
        """
        이 함수는 수정하지 않아도 됩니다.
        likelihood, prior를 활용하여 실제 데이터에 대해 posterior를 구하고 확률로 변환하는 함수.
        """
        posterior = self.get_posterior(x)
        return np.argmax(posterior, axis=1)



