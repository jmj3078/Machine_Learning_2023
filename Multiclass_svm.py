import numpy as np
from scipy.optimize import minimize

class SVM:
    def __init__(self, C=1, max_iter=1000):
        self.C = C
        self.max_iter = max_iter
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
    
    def kernel(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

    def fit(self, X, y):
        n, n_features = X.shape
        
        # gamma값의 계산. RBF kernel, gaussian kernel의 r값은 원래 1/sigma^2으로써, 그대로 이를 계산하여 사용할 수 있다.
        # 다만 분산의 값이 작아질 수록 복잡한 decision boundary를 형성할 수 있고, 함수의 복잡도를 tuning할 수 있기 때문에 
        # 직접 지정하도록 할 수도 있지만 parameter tuning을 진행하지 않았다.
        self.gamma = 1 / (n_features * np.var(X))

        # Objective Function의 정의. sum(α) - 0.5 * sum(i)sum(j)αi*αj*yi*yj*kernel(xi, xj)
        # minimization problem으로 변경하기 위하여 부호를 바꾸어 표기.
        def objective(alphas):
            return 0.5 * np.sum(alphas * alphas * y * y[:, np.newaxis] * np.array([[self.kernel(X[i], X[j]) for j in range(n)] for i in range(n)])) - np.sum(alphas)
       
        # constraints : 최적화 문제에서 사용되는 제약 조건. SVM에서는 0 < sum(α) < C 이어야 한다는 제약 조건.
        def constraint(alphas):
            return np.sum(alphas * y)

        # optimizer를 위한 bound 설정 : α값이 가질 수 있는 범위의 설정. 0부터 C까지의 범위 설정
        # C는 Regularization cost로서 Margin의 넓이와 Margin에서 벗어나는 값들의 Penalty에 대하여 Trade-off를 조절해주는 역할을 함.
        bounds = [(0, self.C) for i in range(n)]

        # optimizer의 초기 α값 설정
        init_guess = np.zeros(n)

        # optimzer의 최대 반복 횟수 설정
        options = {'maxiter': self.max_iter}
        constraints = {'type': 'eq', 'fun': constraint}
        
        # scipy 패키지의 minimize함수 사용하여 optimization진행
        res = minimize(objective, init_guess, bounds=bounds, constraints=constraints, options=options)

        # 최적화된 alpha값을 저장.
        self.alphas = res.x

        # KKT 조건의 고려. 모든 x값에 대하여 y*(w*x+b)-1+ξ≥0, α*ξ= 0 을 만족하고, alpha값이 0과 C사이의 값을 가지도록 한다.
        # 위 조건은 모든 x에 대하여 성립해야 하기 때문에 optimization 과정에서는 모든 x값에 대하여 alpha값을 계산한 다음
        # KKT 조건을 만족하는 alpha값을 추려내고, support vector를 뽑아낸다.
        # 그 다음 support vector만을 이용하여 decision boundary를 계산한다.
        threshold = 1e-5 # threshold의 설정
        self.support_vectors = X[self.alphas > threshold]
        self.support_vector_labels = y[self.alphas > threshold]

    def predict(self, X):
        # 학습된 alpha값을 통해서 input data에 대한 분류 결과 return
        predictions = []
        for sample in X:
            prediction = 0
            for i in range(len(self.support_vectors)):
                prediction += self.alphas[i] * self.support_vector_labels[i] * self.kernel(sample, self.support_vectors[i])
            predictions.append(np.sign(prediction))

        return np.array(predictions)

# 본 코드는 https://hoonst.github.io/2021/01/13/SVM-KKT-Conditions/ 등 다양한 웹 포스트에서 참고한 공식들을 바탕으로 구현되었습니다.
# 정확한 결과가 나오지 않을 수 있으며, 계산 과정에서 정확하지 않거나 구현이 잘못된 부분이 있을 수 있습니다.