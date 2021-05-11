# Light GBM은 leaf-wise 방식을 취하고 있기 때문에 수렴이 굉장히 빠르지만
# , 파라미터 조정에 실패할 경우 과적합을 초래할 수 있다.

import numpy as np
import db_connect as db
import pandas as pd
import warnings
import joblib
import timeit
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_absolute_error as mae_
from sklearn.metrics import mean_squared_error as mse_
from time import time
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor


# db 직접 불러오기 

# 0 없다
# query = "SELECT a.date, IF(DATE LIKE '2019-%', '2019', '2020') AS YEAR , CASE WHEN DATE LIKE '%-01-%' THEN '1' WHEN  DATE LIKE '%-02-%' THEN '2' WHEN  DATE LIKE '%-03-%' THEN '3' WHEN  DATE LIKE '%-04-%' THEN '4'\
# WHEN  DATE LIKE '%-05-%' THEN '5' WHEN  DATE LIKE '%-06-%' THEN '6' WHEN  DATE LIKE '%-07-%' THEN '7' WHEN  DATE LIKE '%-08-%' THEN '8' WHEN  DATE LIKE '%-09-%' THEN '9' WHEN  DATE LIKE '%-10-%' THEN '10' WHEN  DATE LIKE '%-11-%' THEN '11' ELSE '12' END AS MONTH,\
# DAYOFWEEK (DATE)AS DAY, a.time, s.index AS category, d.index AS dong, a.value FROM `business_location_data` AS a INNER JOIN `category_table` AS s ON a.category = s.category INNER JOIN `location_table` AS d ON a.dong = d.location WHERE si = '서울특별시'"

# 0 있다
query = "SELECT * FROM `main_data_table`"

db.cur.execute(query)
dataset = np.array(db.cur.fetchall())


# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)

db.connect.commit()


# train, test 나누기
pred = df.iloc[:,1:-1]

train_value = df[ '2020-09-01' > df['date'] ]

x_train1 = train_value.iloc[:,1:-1].astype('int64').to_numpy()
y_train1 = train_value.iloc[:,-1].astype('int64').to_numpy()

test_value = df[df['date'] >=  '2020-09-01']

x_pred = test_value.iloc[:,1:-1].astype('int64').to_numpy()
y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()

# print(x_train1.shape, y_train1.shape) #(3472128, 6) (3472128,)
# print(x_pred.shape, y_pred.shape)   #(177408, 6) (177408,)


kfold = KFold(n_splits=5, shuffle=True)

# parameters       
parameters = {'max_depth': [5 ,6 ,7 ,10, 15, 20], 'num_leaves':[10,20,30,200], 'n_estimators':[200,300,400],
                'min_child_samples': [20, 40, 60,100,200], 'learning_rate':[0.1,0.001,0.5],
                'subsample': [0.8, 1]}
#참고 : https://greeksharifa.github.io/machine_learning/2019/12/09/Light-GBM/

num = 0 

r2_list = []
rmse_list = []
mae_list = []
time_list = []
best_estimator_list = []

# 훈련 loop
for train_index, valid_index in kfold.split(x_train1):

    # print(train_index, len(train_index))    #2777702
    # print(valid_index, len(valid_index))    #694426

    x_train = x_train1[train_index]
    x_valid = x_train1[valid_index]
    y_train = y_train1[train_index]
    y_valid = y_train1[valid_index]
 
    #2. 모델구성
    model = GridSearchCV(LGBMRegressor(device='gpu'), parameters, cv=kfold)

    start_time = timeit.default_timer()

    #3. 훈련
    model.fit(x_train, y_train, eval_metric='rmse', verbose = True, eval_set=[(x_train, y_train), (x_valid, y_valid)], early_stopping_rounds=20)

    finish_time = timeit.default_timer()
    time = round(finish_time - start_time, 2)
    time_list.append(time)
    print(f'{num}fold time : ', time)

    # best_estimator_
    print('최적의 매개변수 : ', model.best_estimator_) 
    best_estimator_list.append(model.best_estimator_) 

    # 모델저장
    joblib.dump(model.best_estimator_, f'C:/Study/mitzy/data/h5/LGBM_kfold_{num}.pkl')

    # 모델로드
    model = joblib.load(f'C:/Study/mitzy/data/h5/LGBM_kfold_{num}.pkl')

    #4. 평가, 예측
    y_predict = model.predict(x_pred)
    print('예측값 : ', y_predict[:5])
    print('실제값 : ', y_pred[:5])


    # r2_list
    r2 = r2_score(y_pred, y_predict)
    print('r2 score     :', r2)
    r2_list.append(r2)
    # rmse_list
    rmse = mse_(y_pred, y_predict, squared=False)
    print('rmse score     :', rmse)
    rmse_list.append(rmse)
    # mae_list
    mae = mae_(y_pred, y_predict)
    print('mae score     :', mae)
    mae_list.append(mae) 

    num += 1

print('============================================================')
r2_list = np.array(r2_list)
print('r2_list : ', r2_list)
time_list = np.array(time_list)
print('time_list : ', time_list)
rmse_list = np.array(rmse_list)
print('rmse_list : ',rmse_list)
mae_list = np.array(mae_list)
print('mae_list : ',mae_list)
best_estimator_list = np.array(best_estimator_list)
print('best_estimator_list : ',best_estimator_list)

# 이틀동안 1fold도 안 끝남






    