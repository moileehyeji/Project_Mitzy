# 전이학습

import numpy as np
import db_connect as db
import pandas as pd
import timeit
import tensorflow as tf

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
from tensorflow.keras.activations import tanh, relu, elu, selu, swish
from sklearn.metrics import mean_absolute_error as mae_
from sklearn.metrics import mean_squared_error as mse_
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.layers import Embedding, Input, UpSampling2D, GlobalAveragePooling2D, Flatten, Conv2D, Dropout, Dense
from tensorflow.keras.models import Model
import autokeras as ak

# db 직접 불러오기 
fold_score_list = []
history_list=[]


def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

def mish(x):
    return x * K.tanh(K.softplus(x))

def callbacks(modelpath):
    er = EarlyStopping(monitor = 'val_loss',patience=6)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True, verbose=1)
    lr = ReduceLROnPlateau(monitor = 'val_loss',patience=3, factor=0.3, verbose=1)
    return er,mo,lr

def build_model(acti, opti, lr):   

    # # 2. 모델구성
    densenet = DenseNet121(include_top=False, input_shape=(42,42,3), weights='imagenet')
    inputs = Input(shape=(x_train.shape[1],1,1),name='input')
    a = Conv2D(3, kernel_size=(1,1))(inputs)
    a = UpSampling2D(size=(1,42))(a)
    x = densenet(a)
    x = Flatten()(x)
    x = Dense(32, activation=acti)(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    # 3. 컴파일 훈련        
    model.compile(loss='mse', optimizer = opti(learning_rate=lr), metrics='mae')

    return model

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

    return  score_list, y_predict

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
        x = train_value.iloc[:,1:-1].astype('int64')
        y = train_value.iloc[:,-1].astype('int64').to_numpy()
    else:
        test_value = df[df['date'] >=  '2020-09-01']
        x = test_value.iloc[:,1:-1].astype('int64')
        y = test_value.iloc[:,-1].astype('int64').to_numpy()

    
    # 원 핫으로 컬럼 추가해주는 코드!!!!!    
    x = pd.get_dummies(x, columns=["category", "dong"]).to_numpy()
    # 카테고리랑 동만 원핫으로 해준다 

    return x, y

x_train, y_train = load_data("SELECT * FROM main_data_table WHERE (TIME != 2 AND TIME != 3 AND TIME != 4 AND TIME != 5  AND TIME != 6 AND TIME != 7 AND TIME != 8) ORDER BY DATE, YEAR, MONTH ,TIME, category ASC")
x_pred, y_pred = load_data("select * from main_data_table ORDER BY DATE, YEAR, MONTH ,TIME, category ASC",is_train=False)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  train_size=0.9, random_state = 77, shuffle=True ) 

x_train = x_train.reshape(x_train.shape[0], 42,1,1)
x_val = x_val.reshape(x_val.shape[0], 42,1,1)
x_pred = x_pred.reshape(x_pred.shape[0], 42,1,1)

print(x_train.shape, y_train.shape) #(2459424, 42,1,1) (2459424,)
print(x_pred.shape, y_pred.shape)   #(177408, 42,1,1) (177408,)

model = build_model('relu', Adam, 0.01)

modelpath = f'./mitzy/data/modelcheckpoint/22_deltime_data_DenseNet.hdf5'
er,mo,lr = callbacks(modelpath)

model.fit(x_train, y_train, epochs=10, validation_data=(x_val,y_val), callbacks=[er,mo,lr], batch_size = 120)#, callbacks = [es, re, mo]) # validation_splitr 기본 0.2

results, y_predict = evaluate_list(model)
print(results) 

terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time))

# 엑셀 추가 코드 
# 경로 변경 필요!!!!
df = pd.DataFrame(y_predict)
df['test'] = y_pred
df.to_csv('./mitzy/data/csv/22_deltime_data_DenseNet.csv',index=False)


import matplotlib.pyplot as plt 
fig = plt.figure( figsize = (12, 4))
chart = fig.add_subplot(1,1,1)
chart.plot(y_pred, marker='o', color='blue', label='Actual')
chart.plot(y_predict, marker='^', color='red', label='Predict')
plt.legend(loc = 'best') 
plt.show()

# r2 :  -0.001651924140112504
# rmse :  3.8846373013330293
# mae :  1.3491295714046898
# mse :  15.09040696290796
# [-0.001651924140112504, 3.8846373013330293, 1.3491295714046898, 15.09040696290796]
# 18482.583662초 걸렸습니다.