# 완성
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pymysql
import math
from scipy.stats import norm
from scipy.stats import zscore 
from scipy.stats import norm
from scipy import stats

matplotlib.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.family'] = "Malgun Gothic" # 한글 변환


def connect_mysql(query):
    connect = pymysql.connect(host='mitzy.c7xaixb8f0ch.ap-northeast-2.rds.amazonaws.com', user='mitzy', password='mitzy1234!', db='mitzy',\
                            charset='utf8')
    cur = connect.cursor()
    query = query
    
    cur.execute(query)
    select = np.array(cur.fetchall())
    connect.commit()

    return select


#결측값 시각화 함수
def view_missingvalue(df):
    
    df = pd.DataFrame(df)#, columns=['date', 'time', 'category', 'si', 'dong', 'value'])

    # ===========by seaborn
    import seaborn as sns

    # ax = sns.heatmap(df.isnull(), cbar=False)
    # plt.title('sns.heatmap')
    # plt.show()

    # ===========by missingno
    import missingno as msno

    # 1) matrix : 최대 50개의 레이블이 지정된 열만 요약해서 표시
    # ax = msno.matrix(df)
    # plt.title('msno.matrix')
    # plt.show()

    # 2) bar chart : 각열의 결측치가 합해진 값(log=True or False)
    ax = msno.bar(df, log=True)
    plt.title('msno.bar')
    plt.show()

    # 3) heatmap : 결측치가 있는 컬럼만 표시, 상관관계를 파악하기에 효과적 
    # ax = msno.heatmap(df)
    # plt.title('msno.heatmap')
    # plt.show()

    # 4) dendrogram : 결측값이 있는 컬럼의 상관관계를 파악하기에 효과적
    ax = msno.dendrogram(df)
    plt.title('msno.dendrogram')
    plt.show()

    return df

select = connect_mysql("SELECT * FROM `business_location_data`")
# view_missingvalue(df)





# 세종시 결측값 확인하기
df = pd.DataFrame(select, columns=['date', 'time', 'category', 'si', 'dong', 'value'])
print(df.loc[df['si'] == '세종특별자치시'])




# 결측값 삭제함수
# axis = 0 행기준/ 1 열기준
# how = 'any' 하나라도 결측값이면 / 'all' 모두 결측값이면
# thresh = 결측값이 몇개 이상이면 삭제?
# subset = ['컬럼명', '컬럼명'] 특정 컬럼 내의 결측치가 있는 행 삭제
# inplace=True 실제 데이터프레임에 나온 값을 저장할 것인지
# 예) dong컬럼 결측값있는 행 모두 삭제
''' trans1 = df.dropna(axis=0, how = 'any',thresh = 1,subset = ['dong'])
print(trans1.isnull())  
view_missingvalue(trans1)
 '''




# 결측값 채우기 함수

# 1) SimpleImputer
# strategy 옵션
#     'mean': 평균값 (디폴트)
#     'median': 중앙값
#     'most_frequent': 최빈값
#     'constant': 특정값, 예) SimpleImputer(strategy='constant', fill_value=1)
""" from sklearn.impute import SimpleImputer
simpleimputer = SimpleImputer(missing_values=None,strategy='most_frequent')
trans2 = simpleimputer.fit_transform(select)
trans2 = view_missingvalue(trans2)
print(trans2.loc[df['si'] == '세종특별자치시']) """

# 2) df.fillna()
# method 옵션
#       'pad' : 결측치 바로 앞의 값으로 채우기
#       'bfill' : 결측치 바로 뒤의 값으로 채우기
# trans3 = df.fillna(0)
""" trans3 = df.fillna(method='pad')
trans3 = view_missingvalue(trans3)
print(trans3.loc[df['si'] == '세종특별자치시']) """

# 3) interpolate()
# method 옵션
#       default 'linear'
#       'linear': 선형 방법으로 보간
#       'time': 시간/날짜 간격으로 보간. 이때 시간/날짜가 index로 되어있어야함.
#       'index', 'values': ?
#       'pad': 바로 앞에 value 사용
""" ts = pd.Series([1,2,3,np.nan,1000000,4,5])
ts_intp_linear = ts.interpolate(method='polynomial')
print(ts_intp_linear)
# 0          1.0
# 1          2.0
# 2          3.0
# 3     500001.5
# 4    1000000.0
# 5          4.0
# 6          5.0 """


