# One vs. one SVM
import numpy as np

class Multiclass_SVM :
      def __init__(self, kernel='rbf', C=1.0, gamma=1.0, tol=1e-4, max_iter=500):
            self.kernel = kernel
            self.C = C
            self.gamma = gamma
            self.tol = tol
            self.max_iter = max_iter
            
      def fit(self, X, y):
            self.classes = np.unique(y)
            self.n_classes = len(self.classes)
            self.classifiers =[]
        
            # one-hot encoding 진행
            Y_hot = np.zeros((len(y), self.n_classes))
            for i, c in enumerate(self.classes):
                  Y_hot[y == c, i] = 1
            
            # one-vs-one approach, 총 m(m-1)/2 번의 Binary Classification을 진행하고 사용한 모델을 저장한다.
            # voting을 통하여 가장 많이 뽑힌 class로 분류를 진행. 느린 방식이지만 one vs rest approach보다 더 안정적인 성능을 기대할 수 있다.
            for i in range(self.n_classes):
                  for j in range(i+1, self.n_classes):
                        print(f"class {i}, {j} Binary SVM")
                        Xi = X[(y==self.classes[i]) | (y==self.classes[j])] # numpy의 boolean operator "|" 사용하여 multiclass분류 진행
                        Yi = Y_hot[(y == self.classes[i]) | (y == self.classes[j]), :][:, [i,j]]
                        svm = SVM(self.C, self.gamma, self.tol, self.max_iter)
                        svm.fit(Xi, np.argmax(Yi, axis=1) * 2 - 1) # class label을 -1, 1로 변경하고, vector형태로 변경
                        self.classifiers.append((clf, i, j))
      
      def predict(self, X):
            votes = np.zeros((len(X), self.n_classes))
            # 저장해둔 모델을 통해서 prediction을 진행하고, 각 class에 voting 진행.
            for svm, i, j in self.classifiers :
                  pred = svm.predict(X)
                  votes[pred == 1, i] += 1
                  votes[pred == -1, j] += 1
                  
            # 최종적으로 가장 많은 vote가 있는 class로 분류
            return self.classes[np.argmax(votes, axis=1)] 
        
# Binary SVM
class SVM:
    def __init__(self, C=1.0, gamma=1.0, tol=1e-4, max_iter=200, n_sv=10):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.n_sv = 10
             
    def fit(self, X, y):
        self.X = X
        self.y = y
        
        # kernel matrix의 계산
        K = self.gram_matrix(X)
        
        # alpha값과 bias의 초기화. 
        alpha = np.random.rand(len(X))
        b = 0
        self.b = 0
        # support vector의 초기화. SVM에서 alpha값은 0보다 크거나 같다는 제약 조건이 존재한다. 임의의 0과 1사이의 값을 부여
        self.sv_idx = np.where(alpha > 1e-5)[0]
        self.support_vectors = X[self.sv_idx]
        self.support_vector_labels = y[self.sv_idx]
        self.support_vector_weights = alpha[self.sv_idx]
        
        # Optimization 
        for _ in range(self.max_iter):
            alpha_old = np.copy(alpha)
            for i in range(len(X)):
                
                E_i = self.margin(X[i]) - y[i]
                if (y[i]*E_i < -self.tol and alpha[i] < self.C) or (y[i]*E_i > self.tol and alpha[i] > 0):
                    j = np.random.choice([k for k in range(len(X)) if k != i])
                    E_j = self.margin(X[j]) - y[j]
                    
                    alpha_i_old, alpha_j_old = alpha[i], alpha[j]
                    
                    L, H = self._compute_bounds(alpha_i_old, alpha_j_old, y[i], y[j], self.C)
                    if L == H:
                        continue
                    
                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    # alpha값 갱신
                    alpha[j] -= y[j] * (E_i - E_j) / eta
                    alpha[j] = np.clip(alpha[j], L, H)
                    alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])
                    
                    b1 = b - E_i - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                    b2 = b - E_j - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - y[j] * (alpha[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0
                        
            # Convergence 유무 확인. iteration 끝나기 전에 수렴할 시 반복문 종료
            diff = np.linalg.norm(alpha - alpha_old)
            if diff < self.tol:
                print("convergence success")
                break
                
        # 추정한 alpha값을 기반으로 다시 support vector 연산
        self.sv_idx = np.where(alpha > 1e-5)[0]
        self.support_vectors = X[self.sv_idx]
        self.support_vector_labels = y[self.sv_idx]
        self.support_vector_weights = alpha[self.sv_idx]
        
        # 상수항 계산 (beta0)
        self.b = np.mean(self.support_vector_labels - self._predict(self.support_vectors))

    def predict(self, X):
        # Compute the kernel matrix between test data and support vectors
        K = self.gram_matrix(X, self.support_vectors)
        # Compute the predicted labels
        pred = np.dot(K, self.support_vector_weights * self.support_vector_labels)
        
        return np.sign(pred + self.b)
    
    def gram_matrix(self, X, Y=None):
        if Y is None:
            Y = X
        dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2*np.dot(X, Y.T)
        K = np.exp(-self.gamma * dists)
        
        return K

    def margin(self, x):
        # Compute the RBF kernel between x and the support vectors
        K = self._rbf_kernel(x, self.support_vectors, self.gamma)
        
        # Compute the margin
        margin = np.dot(self.support_vector_weights * self.support_vector_labels, K) + self.b
        margin *= self.y[np.argmin(np.abs(self.support_vector_weights))]  # account for sign of y
        return margin
    
    def _rbf_kernel(self, x, X, gamma):
        # Compute the pairwise squared Euclidean distances between x and X
        dists = np.sum((X - x)**2, axis=1)
        # Compute the RBF kernel between x and X
        K = np.exp(-gamma * dists)
        return K
    
    def _compute_bounds(self, alpha_i_old, alpha_j_old, y_i, y_j, C):
        if y_i != y_j:
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(C, C + alpha_j_old - alpha_i_old)
        else:
            L = max(0, alpha_i_old + alpha_j_old - C)
            H = min(C, alpha_i_old + alpha_j_old)
        return L, H