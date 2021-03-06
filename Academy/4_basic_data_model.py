import numpy as np
import db_connect as db
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import  LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor




# db 직접 불러오기 

query = "SELECT a.date, IF(DATE LIKE '2019-%', '2019', '2020') AS YEAR , CASE WHEN DATE LIKE '%-01-%' THEN '1' WHEN  DATE LIKE '%-02-%' THEN '2' WHEN  DATE LIKE '%-03-%' THEN '3' WHEN  DATE LIKE '%-04-%' THEN '4'\
WHEN  DATE LIKE '%-05-%' THEN '5' WHEN  DATE LIKE '%-06-%' THEN '6' WHEN  DATE LIKE '%-07-%' THEN '7' WHEN  DATE LIKE '%-08-%' THEN '8' WHEN  DATE LIKE '%-09-%' THEN '9' WHEN  DATE LIKE '%-10-%' THEN '10' WHEN  DATE LIKE '%-11-%' THEN '11' ELSE '12' END AS MONTH,\
DAYOFWEEK (DATE)AS DAY, a.time, s.index AS category, d.index AS dong, a.value FROM `business_location_data` AS a INNER JOIN `category_table` AS s ON a.category = s.category INNER JOIN `location_table` AS d ON a.dong = d.location WHERE si = '서울특별시'"

db.cur.execute(query)
dataset = np.array(db.cur.fetchall())


# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)

db.connect.commit()


# train, test 나누기

train_value = df[ '2020-09-01' >= df['date'] ]

x_train = train_value.iloc[:,1:-1]
y_train = train_value.iloc[:,-1]

test_value = df[df['date'] >=  '2020-09-01']

x_test = test_value.iloc[:,1:-1]
y_test = test_value.iloc[:,-1]

# print(x_train.shape, y_train.shape) #(321035, 6) (321035,)
# print(x_test.shape, y_test.shape)   #(16707, 6) (16707,)


# 전처리
# x_train, x_val, y_train, y_val = train_test_split(x, y, train_size = 0.8, random_state = 120, shuffle = True)

kfold = KFold(n_splits=5, shuffle=True)

# 훈련 loop
scalers = np.array([MinMaxScaler(), StandardScaler()])
models = np.array([DecisionTreeRegressor(), RandomForestRegressor()])
# , KNeighborsRegressor()

result_list = []
for i in models:
    print(i,'   :')

    #2. 모델구성
    model = i

    scores = cross_val_score(model, x_train, y_train, cv=kfold)
    print('scores : ', scores)

    
    #3. 훈련
    model.fit(x_train, y_train)
    #4.평가, 예측
    y_pred = model.predict(x_test)
    print('예측값 : ', y_pred[:5])
    print('실제값 : ', y_test[:5])

    result = model.score(x_test, y_test)
    print('model.score     :', result)
    result_list.append(result)
    # accuracy_score = accuracy_score(y_test, y_pred)  
    # print('accuracy_score  :', accuracy_score)      #TypeError: 'numpy.float64' object is not callable
    # print('r2_score  :', r2_score(y_test, y_pred))
    
    print('\n')   
 
result_list = np.array(result_list)
print(result_list)

""" 
models = np.array([LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()])
========================================== MinMaxScaler()
LinearRegression()    :
scores :  [0.08774794 0.09477771 0.09439108 0.09136506 0.09179506]
model.score     : 0.09949938779142598


DecisionTreeRegressor()    :
scores :  [0.78756353 0.79449959 0.79006814 0.79555934 0.79971829]
model.score     : 0.8037679539588984


RandomForestRegressor()    :
scores :  [0.80728974 0.81007489 0.80817042 0.81582116 0.8154829 ]
model.score     : 0.8440603312583925


========================================== StandardScaler()
LinearRegression()    :
scores :  [0.09159671 0.09137116 0.09427318 0.09148861 0.09147646]
model.score     : 0.09949938779142964


DecisionTreeRegressor()    :
scores :  [0.79122314 0.79975187 0.78383911 0.79678264 0.78355703]
model.score     : 0.8028220295261673


RandomForestRegressor()    :
scores :  [0.80983662 0.80886558 0.80379471 0.81214155 0.81447037]
model.score     : 0.8442904746075621


[0.09949939 0.80376795 0.84406033 0.09949939 0.80282203 0.84429047] 
"""



