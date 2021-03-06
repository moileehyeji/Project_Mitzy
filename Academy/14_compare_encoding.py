# 덴스회귀 공통된 모델로 인코딩 다르게해서 경과시간 비교해보자

# 비교1 mean encoding

import numpy as np
import db_connect as db
import pandas as pd
import timeit
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv2D, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout,LSTM, Conv2D,Input,Activation

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K

# db 직접 불러오기 

'''
query = "SELECT a.date, IF(DATE LIKE '2019-%', '2019', '2020') AS YEAR , CASE WHEN DATE LIKE '%-01-%' THEN '1' WHEN  DATE LIKE '%-02-%' THEN '2' WHEN  DATE LIKE '%-03-%' THEN '3' WHEN  DATE LIKE '%-04-%' THEN '4'\
WHEN  DATE LIKE '%-05-%' THEN '5' WHEN  DATE LIKE '%-06-%' THEN '6' WHEN  DATE LIKE '%-07-%' THEN '7' WHEN  DATE LIKE '%-08-%' THEN '8' WHEN  DATE LIKE '%-09-%' THEN '9' WHEN  DATE LIKE '%-10-%' THEN '10' WHEN  DATE LIKE '%-11-%' THEN '11' ELSE '12' END AS MONTH,\
DAYOFWEEK (DATE)AS DAY, a.time, s.index AS category, d.index AS dong, a.value FROM `business_location_data` AS a INNER JOIN `category_table` AS s ON a.category = s.category INNER JOIN `location_table` AS d ON a.dong = d.location WHERE si = '서울특별시'"
'''

query = "select * from main_data_table ORDER BY DATE, YEAR, MONTH ,TIME, category ASC "


db.cur.execute(query)
dataset = np.array(db.cur.fetchall())


# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)

db.connect.commit()

df.iloc[:,-1] = df.iloc[:,-1].astype('int32')
df['category'] = df['category'].astype('int32')
df['dong'] = df['dong'].astype('int32')

# #==========mean encodong--=============================
# print(df[['dong']].head())
# print(' ↧ ↧ ↧ ↧ ↧ ↧ ↧')
# target = 'value'
# category_mean = df.groupby('category')[target].mean()
# dong_mean = df.groupby('dong')[target].mean()

# df['category'] = df['category'].map(category_mean)
# df['dong'] = df['dong'].map(dong_mean)

# print(df[['dong']].head())
# #======================================================


# train, test 나누기

train_value = df[ '2020-09-01' > df['date'] ]

x_train = train_value.iloc[:,1:-1].astype('int64').to_numpy()
y_train = train_value.iloc[:,-1].astype('int64').to_numpy()

test_value = df[df['date'] >=  '2020-09-01']

x_pred = test_value.iloc[:,1:-1].astype('int64').to_numpy()
y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()


def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

kfold = KFold(n_splits=3, shuffle=True)

start_time = timeit.default_timer()

num = 0 
r2_list = []
rmse_list = []
loss_list = []

leaky_relu = tf.nn.leaky_relu

def mish(x):
    return x * K.tanh(K.softplus(x))

for train_index, test_index in kfold.split(x_train): 

    x_train1, x_test1 = x_train[train_index], x_train[test_index]
    y_train1, y_test1 = y_train[train_index], y_train[test_index]
    x_train1, x_val, y_train1, y_val = train_test_split(x_train1, y_train1,  train_size=0.9, random_state = 77, shuffle=False ) 

   # 2. 모델구성

    model = Sequential()
    model.add(Dense(1024, activation=leaky_relu ,input_dim= x_train.shape[1]))
    model.add(Dropout(0.2))
    model.add(Dense(256,activation=leaky_relu))
    model.add(Dropout(0.2))
    model.add(Dense(64,activation=leaky_relu))
    model.add(Dense(16,activation=leaky_relu))
    model.add(Dense(1)) 

    # 3. 컴파일 훈련
    modelpath = './mitzy/data/hdf5/14_compare_encoding'+str(num)+'.hdf5'
    es= EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
    cp =ModelCheckpoint(filepath='modelpath', monitor='val_loss', save_best_only=True, mode='auto')

    model.compile(loss='mse', optimizer='RMSprop', metrics='mae')
    model.fit(x_train1, y_train1, epochs=1000, batch_size=64, validation_data=(x_val,y_val), callbacks=[es,reduce_lr,cp] )

    # 4. 평가, 예측
    loss, mae = model.evaluate(x_test1, y_test1, batch_size=64)
    y_predict = model.predict(x_pred)
    print(loss)

    # RMSE 
    print("RMSE : ", RMSE(y_pred, y_predict))

    # R2 만드는 법
    r2 = r2_score(y_pred, y_predict)
    print("R2 : ", r2)

    num += 1

    r2_list.append(r2_score(y_pred, y_predict))
    rmse_list.append(RMSE(y_pred, y_predict))
    loss_list.append(loss)

print("LSTM 윈도우 없음")
print("r2 : ",r2_list)
print("RMSE : ",rmse_list)
print("loss : ",loss_list)

terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time))
