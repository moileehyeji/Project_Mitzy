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
from xgboost import XGBClassifier, XGBRegressor



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
parameters = [
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.3, 0.001, 0.01], 'max_depth': [4,5,6]},
    {'n_estimators':[90, 100, 110], 'learning_rate':[0.1, 0.001, 0.01], 'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1]},
    {'n_estimators':[90,110], 'learning_rate':[0.1, 0.001, 0.5],  'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1], 'colsample_bylevel': [0.6, 0.7, 0.9]}
]
# parameters = [
#     {'n_estimators':[3], 'learning_rate':[0.1], 'max_depth': [4]},
#     {'n_estimators':[3], 'learning_rate':[0.1], 'max_depth':[4], 'colsample_bytree':[0.6]},
#     {'n_estimators':[3], 'learning_rate':[0.1],  'max_depth':[4], 'colsample_bytree':[0.6], 'colsample_bylevel': [0.6]}
# ]

num = 0 

r2_list = []
rmse_list = []
mae_list = []
time_list = []

# 훈련 loop
for train_index, valid_index in kfold.split(x_train1):

    # print(train_index, len(train_index))    #2777702
    # print(valid_index, len(valid_index))    #694426

    x_train = x_train1[train_index]
    x_valid = x_train1[valid_index]
    y_train = y_train1[train_index]
    y_valid = y_train1[valid_index]
 
    #2. 모델구성
    model = GridSearchCV(XGBRegressor(use_label_encoder=False,
                        tree_method = 'gpu_hist',        
                        predictor = 'gpu_predictor',        
                        gpu_id = 0), parameters, cv=kfold)

    start_time = timeit.default_timer()

    #3. 훈련
    model.fit(x_train, y_train, eval_metric='rmse', verbose = True, eval_set=[(x_train, y_train), (x_valid, y_valid)], early_stopping_rounds=20)

    finish_time = timeit.default_timer()
    time = round(finish_time - start_time, 2)
    time_list.append(time)
    print(f'{num}fold time : ', time)

    # best_estimator_
    print('최적의 매개변수 : ', model.best_estimator_)  

    # 모델저장
    joblib.dump(model.best_estimator_, f'C:/Study/mitzy/data/h5/XGB_kfold_{num}.pkl')

    # 모델로드
    model = joblib.load(f'C:/Study/mitzy/data/h5/XGB_kfold_{num}.pkl')

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

r2_list = np.array(r2_list)
print('r2_list : ', r2_list)
time_list = np.array(time_list)
print('time_list : ', time_list)
rmse_list = np.array(rmse_list)
print('rmse_list : ',rmse_list)
mae_list = np.array(mae_list)
print('mae_list : ',mae_list)

# 최적의 매개변수 :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.3, max_delta_step=0, max_depth=6,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=300, n_jobs=8, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', use_label_encoder=False,
#              validate_parameters=1, verbosity=None)
# 예측값 :  [-0.01496865 -0.06798053  0.07910885 -0.092887   -0.01300849]
# 실제값 :  [0 0 0 0 0]
# r2 score     : 0.854149955688163
# rmse score     : 1.482331521683476
# mae score     : 0.4839776794555641
# 파일이름 : XGB_kfold_0(1).pkl



'''
파일이름 : XGB_kfold_0 ~ 4.pkl
4fold time :  7004.64
최적의 매개변수 :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=0,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.3, max_delta_step=0, max_depth=6,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=300, n_jobs=8, num_parallel_tree=1,
             predictor='gpu_predictor', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='gpu_hist', use_label_encoder=False,
             validate_parameters=1, verbosity=None)
예측값 :  [ 0.03414506  0.07111073  0.17881331 -0.00435555  0.11635181]
실제값 :  [0 0 0 0 0]
r2 score     : 0.8499472574457709
rmse score     : 1.5035366886251995
mae score     : 0.4969941971408308
r2_list :  [0.85199009 0.85408845 0.8568749  0.85124046 0.84994726]
time_list :  [7011.22 7176.65 7282.39 7138.7  7004.64]
rmse_list :  [1.49326697 1.48264405 1.46841887 1.4970437  1.50353669]
mae_list :  [0.47429725 0.4809542  0.47178513 0.47795754 0.4969942 ]
'''







    