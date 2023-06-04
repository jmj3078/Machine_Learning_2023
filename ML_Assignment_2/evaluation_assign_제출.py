 # 이름과 학번을 작성해주세요
__author__ = "조명재"
__id__ = "2018312990"

# 넘파이 이외의 어떠한 라이브러리도 호출할 수 없습니다.
import numpy as np

# 아래에 코드를 작성해주세요.

########  Notice! 평가 기준  #########
#  accuracy 함수 : 0.5점             #
#  my_confusion_matrix 함수 : 1점    #
#  recall 함수 : 0.5점               #
#  precision 함수 : 0.5점            #
#  f1 함수 : 0.5점                   #
#  TF-IDF 함수 : 2점                 #
############  총 5점 부여  ###########

class my_evaluation_metrics:
    
    def __init__(self):
        self.author = __author__
        self.id = __id__

    def my_accuracy(self, y_true, y_pred):
        """
        정확도를 계산하는 함수입니다.
        Binary classification은 물론 Multi-label classification에도 적용 가능하도록 구현해주세요.

        y_true : 실제 값입니다.
        y_pred : 예측 값입니다.

        output type은 float 입니다.
        """
        n = y_true.size
        acc = (y_true == y_pred).sum() / n 
        # tuple과 sum을 활용하여 한번에 TP+TN 수를 계산
        # label에 대한 제한이 정해져있지 않기 때문에 Multi-label classification에도 적용가능
        return acc

    def my_confusion_matrix(self, y_true, y_pred, pos_label=1):
        """
        Confusion Matrix를 출력하는 함수입니다.
        True Positive, True Negative, False Positive, False Negative를 위한 조건을 None 부분에 입력해주세요.
        반드시 pos_label 변수를 활용하셔야 합니다.

        pos_label : Positive로 설정할 label (Binary classification에서 일반적으로 1을 뜻함)

        output type은 numpy array 입니다.
        """

        cm_result = [[0, 0], [0, 0]]
        for i, value in enumerate(y_pred):
            if y_true[i] == value and value == pos_label:
                cm_result[1][1] += 1 #tp
            elif y_true[i] != value and value == pos_label:
                cm_result[0][1] += 1 #fp
            elif y_true[i] == value and value != pos_label:
                cm_result[0][0] += 1 #tn
            elif y_true[i] != value and value != pos_label:
                cm_result[1][0] += 1 #fn
    
        return np.array(cm_result)


    def my_recall(self, y_true, y_pred):
        """
        Recall을 출력하는 함수입니다.
        Binary classification에서만 작동하도록 구현하세요.
        
        output type은 float 입니다.
        """
        
        TP = ((y_pred == 1) & (y_true == 1)).sum()
        FN = ((y_pred == 0) & (y_true == 1)).sum()
        return TP/(TP+FN)
        # 조건문과 튜플, 그리고 sum() 함수를 활용해서 계산
        
    def my_precision(self, y_true, y_pred):
        """
        Precision을 출력하는 함수입니다.
        Binary classification에서만 작동하도록 구현하세요.
        
        output type은 float 입니다.
        """
        TP = ((y_pred == 1) & (y_true == 1)).sum()
        FP = ((y_pred == 1) & (y_true == 0)).sum()
        return TP / (TP+FP)
        # 조건문과 튜플, 그리고 sum() 함수를 활용해서 계산


    def my_f1(self, y_true, y_pred):
        """
        F1 score를 출력하는 함수입니다.
        Binary classification에서만 작동하도록 구현하세요.
        
        output type은 float 입니다.
        """
        prec = self.my_precision(y_true, y_pred)
        recall = self.my_recall(y_true, y_pred)

        return 2.0/((1/prec) + (1/recall))
        # my_precision 함수와 my_recall 함수를 self.로 불러와서 사용, 조화평균 계산


    def my_tf_idf(self, documents):
        """
        TF-IDF를 출력하는 함수입니다.
        교안을 참고하여 tf_idf 변수에 적합한 값을 입력하세요.
        tf_idf의 shape은 (len(documents), len(word_list))임을 유의하세요.
        """

        # 전체 documents에서 등장하는 word의 리스트입니다.
        word_list = [] 
        for doc in documents:
            splited = doc.split(' ')
            for word in splited:
                if word not in word_list:
                    word_list.append(word)

        # TF-IDF를 연산하기 위해 numpy array를 초기화합니다.
        tf_idf = np.zeros((len(documents), len(word_list)))

        ## fill the blank
        for i, doc in enumerate(documents):
            for j, word in enumerate(word_list):
                tf_idf[i, j] = doc.count(word)/len(doc) #tf값 계산, 행렬에 추가

        idf = np.log(len(documents)/(1+np.count_nonzero(tf_idf, axis=0)))
        # idf는 열 별로 계산할 수 있기 때문에, tf값 만을 tf_idf matrix에 넣어두고 idf를 계산한 뒤 곱함
        # tf값이 0인 doc은 그 단어가 출현하지 않았다는 뜻이기 때문에, count_nonzero함수를 사용해 0이 아닌 문서의 수를 counting
        tf_idf *= idf
        tf_idf
        return tf_idf







