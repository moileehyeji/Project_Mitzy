# 데이터 0full, 0del, time(2-8)del 비교하기

""" import numpy as np
import db_connect as db
import pandas as pd
import timeit
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv2D, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout,LSTM, Conv2D,Input,Activation

from sklearn.metrics import mean_squared_error as mse_
from sklearn.metrics import mean_absolute_error as mae_
from sklearn.metrics import r2_score
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K

def load_data(query):
    query = query
    db.cur.execute(query)
    dataset = np.array(db.cur.fetchall())

    # pandas 넣기
    column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']
    df = pd.DataFrame(dataset, columns=column_name)
    db.connect.commit()

    # train, test 나누기
    pred = df.iloc[:,1:-1]
    train_value = df[ '2020-09-01' > df['date'] ]
    x_train = train_value.iloc[:,1:-1].astype('int64')
    y_train = train_value.iloc[:,-1].astype('int64').to_numpy()

    test_value = df[df['date'] >=  '2020-09-01']
    x_pred = test_value.iloc[:,1:-1].astype('int64')
    y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()

    
    # 원 핫으로 컬럼 추가해주는 코드!!!!!    
    x_train = pd.get_dummies(x_train, columns=["category", "dong"]).to_numpy()
    x_pred = pd.get_dummies(x_pred, columns=["category", "dong"]).to_numpy()
    # 카테고리랑 동만 원핫으로 해준다 

    return x_train, y_train, x_pred, y_pred

def evaluate_list(model):
    score_list = []
    #4. 평가, 예측
    y_predict = model.predict(x_pred)

    # r2_list
    r2 = r2_score(y_pred, y_predict)
    score_list.append(r2)
    print('r2 : ', r2)
    # rmse_list
    rmse = mse_(y_pred, y_predict, squared=False)
    score_list.append(rmse)
    print('rmse : ', rmse)
    # mae_list
    mae = mae_(y_pred, y_predict)
    score_list.append(mae)
    print('mae : ', mae)
    # mse_list
    mse = mse_(y_pred, y_predict, squared=True)
    score_list.append(mse)
    print('mse : ', mse)

    return  score_list

def build_model(acti, opti, lr):   

    # 2. 모델구성
    inputs = Input(shape = (x_train.shape[1],),name = 'input')
    x = Dense(filters=1024, activation=acti)(inputs)
    x = Dense(256, activation=acti)(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation=acti)(x)
    x = Dense(16, activation=acti)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs,outputs=outputs)

    # 3. 컴파일 훈련        
    model.compile(loss='mse', optimizer = opti(learning_rate=lr), metrics='mae')

    return model

def callbacks(modelpath):
    er = EarlyStopping(monitor = 'val_loss',patience=20)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True, verbose=1)
    lr = ReduceLROnPlateau(monitor = 'val_loss',patience=10, factor=0.3, verbose=1)
    return er,mo,lr

def mish(x):
    return x * K.tanh(K.softplus(x))

leaky_relu = tf.nn.leaky_relu


x_train, y_train, x_pred, y_pred = load_data('SELECT * FROM main_data_table WHERE VALUE != 0')

print(x_train.shape, y_train.shape) #(320210, 42) (320210,)
print(x_pred.shape, y_pred.shape)   #(16707, 36) (16707,) 

model = build_model(leaky_relu, Adam, 1e-3) """

import imp
import numpy as np
import db_connect as db
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.layers import Dense, Input, LSTM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB4, EfficientNetB2, EfficientNetB7, VGG16, MobileNet, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation, Dropout, UpSampling2D, Conv2D

# db 직접 불러오기 


# 0 없다

# 0 있다
query = "SELECT * FROM main_data_table WHERE VALUE != 0"
query1 = "select * from main_data_table ORDER BY DATE, YEAR, MONTH ,TIME, category ASC"
db.cur.execute(query)
dataset = np.array(db.cur.fetchall())
db.cur.execute(query1)
dataset1 = np.array(db.cur.fetchall())

# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)
df1 = pd.DataFrame(dataset1, columns=column_name)

db.connect.commit()

train_value = df[ '2020-09-01' > df['date'] ]

x_train = train_value.iloc[:,1:-1].astype('int64')
y_train = train_value['value'].astype('int64').to_numpy()

test_value = df1[df1['date'] >=  '2020-09-01']

x_pred = test_value.iloc[:,1:-1].astype('int64')
y_pred = test_value['value'].astype('int64').to_numpy()

x_train = pd.get_dummies(x_train, columns=["category", "dong"]).to_numpy()
x_pred = pd.get_dummies(x_pred, columns=["category", "dong"]).to_numpy()

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  train_size=0.9, random_state = 77, shuffle=True ) 
print(x_train.shape, x_val.shape, x_pred.shape) # (3124915, 42) (347213, 42) (177408, 42)

leaky_relu = tf.nn.leaky_relu

inputs = Input(shape=(x_train.shape[1]),name='input')
x = Dense(1024,activation=leaky_relu)(inputs)
x = Dropout(0.2)(x)
x = Dense(256,activation=leaky_relu)(x)
x = Dropout(0.2)(x)
x = Dense(64,activation=leaky_relu)(x)
x = Dense(16,activation=leaky_relu)(x)
outputs = Dense(1)(x)
model = Model(inputs=inputs, outputs=outputs)
model.summary()


es= EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
cp = ModelCheckpoint('./data/hdf5/1_0del.hdf5', monitor='val_loss', save_best_only=True, verbose=1,mode='auto')
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=50, batch_size=1500, validation_data=(x_val,y_val), callbacks=[reduce_lr,cp] )

# from tensorflow.keras.models import load_model
# model = load_model('./data/hdf5/1_0del.hdf5', custom_objects={'leaky_relu': tf.nn.leaky_relu})

# 4. 평가, 예측
loss, mae = model.evaluate(x_pred, y_pred, batch_size=1024)
y_predict = model.predict(x_pred)

from sklearn.metrics import mean_squared_error as mse_
from sklearn.metrics import mean_absolute_error as mae_
# r2_list
r2 = r2_score(y_pred, y_predict)
print('r2 : ', r2)
# rmse_list
rmse = mse_(y_pred, y_predict, squared=False)
print('rmse : ', rmse)
# mae_list
mae = mae_(y_pred, y_predict)
print('mae : ', mae)
# mse_list
mse = mse_(y_pred, y_predict, squared=True)
print('mse : ', mse)

import matplotlib.pyplot as plt
 
fig = plt.figure( figsize = (12, 4))
chart = fig.add_subplot(1,1,1)
chart.plot(y_pred, marker='o', color='blue', label='실제값')
chart.plot(y_predict, marker='^', color='red', label='예측값')
plt.legend(loc = 'best') 
# plt.show()

# mse, mae:  15.643658638000488
# RMSE :  3.955227969371693
# R2 :  -0.03838622416290849