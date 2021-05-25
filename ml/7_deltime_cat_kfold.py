# [CatBoost]
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
import tensorflow.keras.backend as K
from time import time
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor


# db 직접 불러오기 
fold_score_list = []
history_list=[]
csv_list = []

def RMSE(y_test, y_predict): 
    return np.sqrt(mse_(y_test, y_predict)) 

start_time = timeit.default_timer()


def load_data(query, is_train = True):
    query = query
    db.cur.execute(query)
    dataset = np.array(db.cur.fetchall())

    # pandas 넣기
    column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']
    df = pd.DataFrame(dataset, columns=column_name)
    db.connect.commit()

    # pred = df.iloc[:,1:-1]

    if is_train == True:
        # train, test 나누기
        train_value = df[ '2020-09-01' > df['date'] ]
        x = train_value.iloc[:,1:-1].astype('float64')
        y = train_value.iloc[:,-1].astype('float64').to_numpy()
    else:
        test_value = df[df['date'] >=  '2020-09-01']
        x = test_value.iloc[:,1:-1].astype('float64')
        y = test_value.iloc[:,-1].astype('float64').to_numpy()

    
    # 원 핫으로 컬럼 추가해주는 코드!!!!!    
    x = pd.get_dummies(x, columns=["category", "dong"]).to_numpy()
    # 카테고리랑 동만 원핫으로 해준다 

    return x, y


x_train1, y_train1 = load_data("SELECT * FROM main_data_table WHERE (TIME != 2 AND TIME != 3 AND TIME != 4 AND TIME != 5  AND TIME != 6 AND TIME != 7 AND TIME != 8) ORDER BY DATE, YEAR, MONTH ,TIME, category ASC")
x_pred, y_pred = load_data("select * from main_data_table ORDER BY DATE, YEAR, MONTH ,TIME, category ASC",is_train=False)

kfold = KFold(n_splits=5, shuffle=True)


r2_list = []
rmse_list = []
mae_list = []
time_list = []
best_estimator_list=[]

num = 0

parameters = {'depth':[4,6,10],'iterations':[250,500],'learning_rate':[0.001,0.01],
              'l2_leaf_reg':[3,5,10],'random_strength':[0.5,0.8] }
# [출처]https://stackoverflow.com/questions/60648547/how-to-increase-accuracy-of-model-using-catboost
#   https://dailyheumsi.tistory.com/136

# 훈련 loop
for train_index, valid_index in kfold.split(x_train1):

    x_train = x_train1[train_index]
    x_valid = x_train1[valid_index]
    y_train = y_train1[train_index]
    y_valid = y_train1[valid_index]
 
    #2. 모델구성
    model = GridSearchCV(CatBoostRegressor(iterations=300,
                        loss_function='RMSE',
                        task_type="GPU",
                        devices='0:1'), parameters, cv=kfold)

    start_time = timeit.default_timer()

    #3. 훈련
    model.fit(x_train, y_train, verbose = True, eval_set=(x_valid, y_valid))

    finish_time = timeit.default_timer()
    time = round(finish_time - start_time, 2)
    time_list.append(time)
    print(f'{num}fold time : ', time)

    # best_estimator_
    print('최적의 매개변수 : ', model.best_estimator_) 
    best_estimator_list.append(model.best_estimator_)  

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


model = CatBoostRegressor(iterations=300,learning_rate=0.1,loss_function='RMSE',
                        depth=4,task_type="GPU",devices='0:1')


