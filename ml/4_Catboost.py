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
from catboost import CatBoostRegressor


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


r2_list = []
rmse_list = []
mae_list = []
time_list = []

num = 0

# 훈련 loop
for train_index, valid_index in kfold.split(x_train1):

    # print(train_index, len(train_index))    #2777702
    # print(valid_index, len(valid_index))    #694426

    x_train = x_train1[train_index]
    x_valid = x_train1[valid_index]
    y_train = y_train1[train_index]
    y_valid = y_train1[valid_index]
 
    #2. 모델구성
    model = CatBoostRegressor(iterations=300,
                        learning_rate=0.1,
                        loss_function='RMSE',
                        depth=4,
                        task_type="GPU",
                        devices='0:1')

    start_time = timeit.default_timer()

    #3. 훈련
    model.fit(x_train, y_train, verbose = True, eval_set=(x_valid, y_valid))

    finish_time = timeit.default_timer()
    time = round(finish_time - start_time, 2)
    time_list.append(time)
    print(f'{num}fold time : ', time)
 

    # 모델저장
    joblib.dump(model, f'./data/h5/cat_kfold_{num}.pkl')

    # 모델로드
    model = joblib.load(f'./data/h5/cat_kfold_{num}.pkl')

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

print('======================================')
r2_list = np.array(r2_list)
print('r2_list : ', r2_list)
rmse_list = np.array(rmse_list)
print('rmse_list : ',rmse_list)
mae_list = np.array(mae_list)
print('mae_list : ',mae_list)
time_list = np.array(time_list)
print('time_list : ', time_list)



# bestTest = 2.018567884
# bestIteration = 269
# Shrink model to first 270 iterations.
# 0fold time :  109.37
# 예측값 :  [0.02546824 0.04774926 0.02546824 0.06957969 0.02123013]
# 실제값 :  [0 0 0 0 0]
# r2 score     : 0.6861136484664634
# rmse score     : 2.1745944598924534
# mae score     : 0.6936731487076918

# bestTest = 1.999693248
# bestIteration = 234
# Shrink model to first 235 iterations.
# 1fold time :  15.33
# 예측값 :  [ 0.03699853  0.06672274  0.03699853 -0.03820688  0.03631691]
# 실제값 :  [0 0 0 0 0]
# r2 score     : 0.6900346398208477
# rmse score     : 2.1609695258951045
# mae score     : 0.6931983078455713

# bestTest = 2.013068492
# bestIteration = 298
# Shrink model to first 299 iterations.
# 2fold time :  15.52
# 예측값 :  [ 0.02265429  0.05351238  0.02265429 -0.00880918  0.02897823]
# 실제값 :  [0 0 0 0 0]
# r2 score     : 0.6948967067197231
# rmse score     : 2.143954226747351
# mae score     : 0.6826225743334872

# bestTest = 2.013494186
# bestIteration = 289
# Shrink model to first 290 iterations.
# 3fold time :  15.35
# 예측값 :  [ 0.01786868  0.03990804  0.01786868 -0.0753282   0.02577407]
# 실제값 :  [0 0 0 0 0]
# r2 score     : 0.6923730600186064
# rmse score     : 2.1528027721787906
# mae score     : 0.6956327232115683
    
# bestTest = 2.048497131
# bestIteration = 299
# 4fold time :  15.52
# 예측값 :  [-0.00998073  0.01145783 -0.00998073 -0.08878739 -0.00019393]
# 실제값 :  [0 0 0 0 0]
# r2 score     : 0.6901867842873692
# rmse score     : 2.1604391119317117
# mae score     : 0.6646526616606021

# r2_list :  [0.68611365 0.69003464 0.69489671 0.69237306 0.69018678]
# rmse_list :  [2.17459446 2.16096953 2.14395423 2.15280277 2.16043911]
# mae_list :  [0.69367315 0.69319831 0.68262257 0.69563272 0.66465266]
# time_list :  [109.37  15.33  15.52  15.35  15.52]

# 파일이름 : cat_kfold_0~4.pkl