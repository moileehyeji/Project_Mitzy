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

def load_data():
    query = "select * from main_data_table ORDER BY DATE, YEAR, MONTH ,TIME, category ASC "
    db.cur.execute(query)
    dataset = np.array(db.cur.fetchall())

    # pandas 넣기
    column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']
    df = pd.DataFrame(dataset, columns=column_name)
    db.connect.commit()

    # 원 핫으로 컬럼 추가해주는 코드!!!!!
    df = pd.get_dummies(df, columns=["category", "dong"])
    # 카테고리랑 동만 원핫으로 해준다 

    # train, test 나누기
    pred = df.iloc[:,1:-1]
    train_value = df[ '2020-09-01' > df['date'] ]
    x_train = train_value.iloc[:,1:-1].astype('int64').to_numpy()
    y_train = train_value.iloc[:,-1].astype('int64').to_numpy()

    test_value = df[df['date'] >=  '2020-09-01']
    x_pred = test_value.iloc[:,1:-1].astype('int64').to_numpy()
    y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()

    return x_train, y_train, x_pred, y_pred


x_train1, y_train1, x_pred, y_pred = load_data()

print(x_train1.shape, y_train1.shape)
print(x_pred.shape, y_pred.shape)

kfold = KFold(n_splits=5, shuffle=True)

# parameters       
parameters = {'max_depth': [9,7], 'num_leaves':[140, 550], 'n_estimators':[500,1000],
                'min_child_samples': [100,200], 'learning_rate':[0.001, 0.0001],
                'subsample': [0.8,1]}
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
    joblib.dump(model.best_estimator_, f'C:/Study/mitzy/data/h5/18_onthot_LGBM_kfold_{num}.pkl')

    # 모델로드
    model = joblib.load(f'C:/Study/mitzy/data/h5/18_onthot_LGBM_kfold_{num}.pkl')

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



'''
r2_list :  [0.08068742 0.08087919 0.0807276  0.08074736 0.08066575]
time_list :  [1246.22 1241.16 1247.2  1260.4  1269.32]
rmse_list :  [0.19971863 0.19969779 0.19971426 0.19971211 0.19972098]
mae_list :  [0.08275147 0.08282184 0.08271877 0.08273623 0.08264006]
best_estimator_list :  [LGBMRegressor(device='gpu', learning_rate=0.001, max_depth=7,
              min_child_samples=100, n_estimators=300, num_leaves=90,
              subsample=0.8)
 LGBMRegressor(device='gpu', learning_rate=0.001, max_depth=7,
              min_child_samples=100, n_estimators=300, num_leaves=30,
              subsample=0.8)
 LGBMRegressor(device='gpu', learning_rate=0.001, max_depth=7,
              min_child_samples=100, n_estimators=300, num_leaves=90,
              subsample=0.8)
 LGBMRegressor(device='gpu', learning_rate=0.001, max_depth=7,
              min_child_samples=100, n_estimators=300, num_leaves=90,
              subsample=0.8)
 LGBMRegressor(device='gpu', learning_rate=0.001, max_depth=7,
              min_child_samples=200, n_estimators=300, num_leaves=30,
              subsample=0.8)]
'''

'''
r2_list :  [0.22134275 0.2210989  0.22112124 0.22125773 0.22162279]
time_list :  [19141.95 17793.29 17607.48 17684.44 17789.44]
rmse_list :  [0.18380619 0.18383497 0.18383233 0.18381622 0.18377313]
mae_list :  [0.07497489 0.07493279 0.07498657 0.07499267 0.0750545 ]
best_estimator_list :  [LGBMRegressor(device='gpu', learning_rate=0.001, max_depth=9,
              min_child_samples=100, n_estimators=1000, num_leaves=140,
              subsample=0.8)
 LGBMRegressor(device='gpu', learning_rate=0.001, max_depth=9,
              min_child_samples=100, n_estimators=1000, num_leaves=550,
              subsample=0.8)
 LGBMRegressor(device='gpu', learning_rate=0.001, max_depth=9,
              min_child_samples=200, n_estimators=1000, num_leaves=550,
              subsample=1)
 LGBMRegressor(device='gpu', learning_rate=0.001, max_depth=9,
              min_child_samples=200, n_estimators=1000, num_leaves=550,
              subsample=0.8)
 LGBMRegressor(device='gpu', learning_rate=0.001, max_depth=9,
              min_child_samples=100, n_estimators=1000, num_leaves=140,
              subsample=1)]
'''



    