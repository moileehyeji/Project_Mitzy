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
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor
from tensorflow.keras.callbacks import ModelCheckpoint



# db 직접 불러오기 

# query = "SELECT a.date, IF(DATE LIKE '2019-%', '2019', '2020') AS YEAR , CASE WHEN DATE LIKE '%-01-%' THEN '1' WHEN  DATE LIKE '%-02-%' THEN '2' WHEN  DATE LIKE '%-03-%' THEN '3' WHEN  DATE LIKE '%-04-%' THEN '4'\
# WHEN  DATE LIKE '%-05-%' THEN '5' WHEN  DATE LIKE '%-06-%' THEN '6' WHEN  DATE LIKE '%-07-%' THEN '7' WHEN  DATE LIKE '%-08-%' THEN '8' WHEN  DATE LIKE '%-09-%' THEN '9' WHEN  DATE LIKE '%-10-%' THEN '10' WHEN  DATE LIKE '%-11-%' THEN '11' ELSE '12' END AS MONTH,\
# DAYOFWEEK (DATE)AS DAY, a.time, s.index AS category, d.index AS dong, a.value FROM `business_location_data` AS a INNER JOIN `category_table` AS s ON a.category = s.category INNER JOIN `location_table` AS d ON a.dong = d.location WHERE si = '서울특별시'"
query = "SELECT * FROM `main_data_table`"

db.cur.execute(query)
dataset = np.array(db.cur.fetchall())


# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)

db.connect.commit()


# train, test 나누기

pred = df.iloc[:,1:-1]

train_value = df[ '2020-09-01' >= df['date'] ]

x_train = train_value.iloc[:,1:-1]
y_train = train_value.iloc[:,-1]

test_value = df[df['date'] >=  '2020-09-01']

x_test = test_value.iloc[:,1:-1]
y_test = test_value.iloc[:,-1]

# print(x_train.shape, y_train.shape) #(321035, 6) (321035,)
# print(x_test.shape, y_test.shape)   #(16707, 6) (16707,)


kfold = KFold(n_splits=5, shuffle=True)

# 훈련 loop
scalers = np.array([MinMaxScaler(), StandardScaler()])
models = np.array([DecisionTreeRegressor(), RandomForestRegressor(), BaggingRegressor(),\
            ExtraTreeRegressor(), ExtraTreesRegressor()])
# , KNeighborsRegressor()
# BaggingRegressor, DecisionTreeRegressor, ExtraTreeRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

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

    # csv생성
    pred_to_csv = model.predict(pred)
    pred_to_csv = pred_to_csv.reshape(-1,1)
    sub = pd.read_csv('./mitzy/data/maindata.csv',sep='\t', header=None)
    sub[str(i)] = pred_to_csv
    # sub.to_csv('C:/Study/lotte/data/15_relu.csv',index=False)
    sub.to_csv(f'./mitzy/data/maindata.csv',index=False)
    
    
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
scores :  [0.78756353 0.79449959 0.79006814 0.79555934 0.79971829]**********
model.score     : 0.8037679539588984


RandomForestRegressor()    :
scores :  [0.80728974 0.81007489 0.80817042 0.81582116 0.8154829 ]**********
model.score     : 0.8440603312583925


========================================== StandardScaler()
LinearRegression()    :
scores :  [0.09159671 0.09137116 0.09427318 0.09148861 0.09147646]
model.score     : 0.09949938779142964


DecisionTreeRegressor()    :
scores :  [0.79122314 0.79975187 0.78383911 0.79678264 0.78355703]*********
model.score     : 0.8028220295261673


RandomForestRegressor()    :
scores :  [0.80983662 0.80886558 0.80379471 0.81214155 0.81447037]*********
model.score     : 0.8442904746075621

[0.09949939 0.80376795 0.84406033 0.09949939 0.80282203 0.84429047] 
"""
#----> scaling 제거, DecisionTreeRegressor, RandomForestRegressor 채택
#===========================================================================

''' DecisionTreeRegressor()    :
scores :  [0.79748968 0.78604755 0.80022464 0.79711822 0.78490801]
예측값 :  [ 7.  2.  2. 10.  3.]
실제값 :  320210     7
320211     2
320212     2
320213    10
320214     3
Name: value, dtype: object
model.score     : 0.8031921561688294


RandomForestRegressor()    :
scores :  [0.80903274 0.80807178 0.81261304 0.81531161 0.80905613]
예측값 :  [5.00371429 2.02835714 1.87466667 7.66478571 3.32295238]
실제값 :  320210     7
320211     2
320212     2
320213    10
320214     3
Name: value, dtype: object
model.score     : 0.8454838137191889


BaggingRegressor()    :
scores :  [0.81260633 0.80060108 0.81179622 0.80852481 0.79527478]
예측값 :  [5.4        2.1        1.6        5.46666667 3.21666667]
실제값 :  320210     7
320211     2
320212     2
320213    10
320214     3
Name: value, dtype: object
model.score     : 0.8373437956246563


ExtraTreeRegressor()    :
scores :  [0.79060942 0.78506496 0.80080387 0.79124999 0.79055212]
예측값 :  [ 7.  2.  2. 10.  3.]
실제값 :  320210     7
320211     2
320212     2
320213    10
320214     3
Name: value, dtype: object
model.score     : 0.811989326295944


ExtraTreesRegressor()    :
scores :  [0.78610909 0.79559382 0.79468896 0.79323791 0.79979382]
예측값 :  [ 7.  2.  2. 10.  3.]
실제값 :  320210     7
320211     2
320212     2
320213    10
320214     3
Name: value, dtype: object
model.score     : 0.8396423544993135


[0.80319216 0.84548381 0.8373438  0.81198933 0.83964235] '''

#===============================================================
# query = "SELECT * FROM `main_data_table`"
''' DecisionTreeRegressor()    :
scores :  [0.82395839 0.82660187 0.82288543 0.82471463 0.82435607]
예측값 :  [0. 0. 0. 0. 0.]
실제값 :  3472128    0
3472129    0
3472130    0
3472131    0
3472132    0
Name: value, dtype: object
model.score     : 0.8361269312718553


RandomForestRegressor()    :
scores :  [0.83216739 0.83234572 0.84503682 0.83451608 0.83689151]
예측값 :  [0. 0. 0. 0. 0.]
실제값 :  3472128    0
3472129    0
3472130    0
3472131    0
3472132    0
Name: value, dtype: object
model.score     : 0.8678614936024045


BaggingRegressor()    :
scores :  [0.83111709 0.83006211 0.83460234 0.82568366 0.83343812]
예측값 :  [0. 0. 0. 0. 0.]
실제값 :  3472128    0
3472129    0
3472130    0
3472131    0
3472132    0
Name: value, dtype: object
model.score     : 0.862727734027764


ExtraTreeRegressor()    :
scores :  [0.82179813 0.82470486 0.82056876 0.8280417  0.82055372]
예측값 :  [0. 0. 0. 0. 0.]
실제값 :  3472128    0
3472129    0
3472130    0
3472131    0
3472132    0
Name: value, dtype: object
model.score     : 0.8588913498708509


ExtraTreesRegressor()    :
scores :  [0.82376855 0.81883197 0.83095461 0.8209106  0.82123834]
예측값 :  [0. 0. 0. 0. 0.]
실제값 :  3472128    0    
3472129    0
3472130    0
3472131    0
3472132    0
Name: value, dtype: object
model.score     : 0.8756960524802946


[0.83612693 0.86786149 0.86272773 0.85889135 0.87569605] '''



