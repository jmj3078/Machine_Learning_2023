# 이름과 학번을 작성해주세요
__author__ = "조명재"
__id__ = "2018312990"

import numpy as np
import pandas as pd


class level_test():
    
    def __init__(self):
    
        self.author = __author__
        self.id = __id__


    def test1(self):
        """
        0부터 100까지의 숫자를 모두 더하는 for문을 구현하세요.
        최종적으로 모두 더한 결과값이 'temp' 변수에 할당하면 됩니다.
        None 부분에 내용을 채워 넣어주세요.

        """

        temp = 0

        for i in range(1, 100):
            temp += i
        
        return temp


    def test2(self, value):
        """
        input으로 들어오는 'value' 변수가
        3.0 미만이면 0, 
        3.0 이상 / 3.2 미만이면 1,
        3.2 이상 / 3.4 미만이면 2,
        위 3가지 경우가 아닌 나머지 경우에는 3을 'result' 변수에 할당하세요.
        None 부분에 내용을 채워 넣으시면 됩니다.

        """

        result = 0
        
        value = input()
        if value < 3.0 :
            result = 0
        elif value < 3.2 :
            result = 1
        elif value < 3.4 :
            result = 2
        else :
            result = 3
        return result


    def test3(self, original_list):
        """
        input으로 들어오는 original_list는 길이가 150인 1차원의 리스트입니다.
        해당 리스트에서 index 50부터 index 100까지 짝수번째(50, 52, 54, ..., 100) index만으로 구성된 리스트를 'result_list' 변수에 할당하세요.
        None 부분에 내용을 채워 넣으시면 됩니다.
            
        """
        temp = original_list[49:100]
        result_list = temp[0::2]

        return result_list


    def test4(self, original_array):
        """
        input으로 들어오는 original_array는 [150,4]의 형태를 가진 2차원의 배열입니다.
        해당 배열에 대해 행 방향으로 평균을 취해 구한 배열 값을 'result_array' 변수에 할당하세요.
        None 부분에 내용을 채워 넣으면 됩니다.

        """

        result_array = original_array.mean(axis="columns")

        return result_array



    def test5(self, df):
        """
        input으로 들어오는 df는 iris 데이터 파일을 불러온 데이터프레임입니다.
        'sepal length (cm)'라는 column에 대한 값이 5 미만인 행만 남겨 'low5_df' 변수에 할당하세요.
        None 부분에 내용을 채워 넣으면 됩니다.
        
        """
        low5_df = df[df["sepal length (cm)"] < 5]

        return low5_df.index.tolist()






