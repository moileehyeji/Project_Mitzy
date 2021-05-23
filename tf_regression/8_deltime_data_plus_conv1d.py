# 2-8삭제 데이터 + 날씨 + 코로나
# conv1d 파라미터 비교

import numpy as np
import db_connect as db
import pandas as pd
import timeit
import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout,LSTM, Conv2D,Input,Activation, LSTM, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model, load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
from tensorflow.keras.activations import tanh, relu, elu, selu, swish
from sklearn.metrics import mean_absolute_error as mae_
from sklearn.metrics import mean_squared_error as mse_
import tensorflow.keras.backend as K

# db 직접 불러오기 
fold_score_list = []
history_list=[]
csv_list = []

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

def mish(x):
    return x * K.tanh(K.softplus(x))

def callbacks(modelpath):
    er = EarlyStopping(monitor = 'val_loss',patience=20)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True, verbose=1)
    lr = ReduceLROnPlateau(monitor = 'val_loss',patience=10, factor=0.3, verbose=1)
    return er,mo,lr

def build_model(acti, opti, lr):   

    # 2. 모델구성
    
     # 2. 모델구성
    inputs = Input(shape = (x_train.shape[1],1),name = 'input')
    # 모델2
    x = Conv1D(filters=32,kernel_size=2,padding='same',strides = 2, activation=acti)(inputs)
    x = Flatten()(x)
    x = Dense(32, activation=acti)(x)
    x = Dense(16, activation=acti)(x)
    x = Dense(8, activation=acti)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs,outputs=outputs)

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
    column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong','temperature','rain','wind','humidity','person', 'value']
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

x_pred, y_pred = load_data("SELECT d.date,YEAR,MONTH, d.day, d.time, category, dong,temperature,rain,wind,humidity, IFNULL( c_person,0) AS person, VALUE FROM main_data_table AS d INNER JOIN `weather` AS s ON d.date = s.DATE AND d.time = s.time LEFT JOIN `covid19_re` AS c ON c.date = d.date ", is_train = False)
x_train, y_train = load_data("SELECT d.date,YEAR,MONTH, d.day, d.time, category, dong,temperature,rain,wind,humidity, IFNULL( c_person,0) AS person, VALUE FROM main_data_table AS d INNER JOIN `weather` AS s ON d.date = s.DATE AND d.time = s.time LEFT JOIN `covid19_re` AS c ON c.date = d.date WHERE (d.TIME != 2 AND d.TIME != 3 AND d.TIME != 4 AND d.TIME != 5  AND d.TIME != 6 AND d.TIME != 7 AND d.TIME != 8) ORDER BY DATE, YEAR, MONTH, DAY, TIME, category, dong ASC")

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1],1)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  train_size=0.9, random_state = 77, shuffle=True ) 

print(x_train.shape, y_train.shape) #(2213481, 46, 1) (2213481,)
print(x_pred.shape, y_pred.shape)   #(177408, 46, 1) (177408,)
print(x_val.shape, y_val.shape)     #(245943, 46, 1) (245943,) 

leaky_relu = tf.nn.leaky_relu
acti_list = [leaky_relu, mish, 'swish', 'elu', 'relu', 'selu','tanh']
opti_list = [RMSprop, Nadam, Adam, Adadelta, Adamax, Adagrad]
batch = 500
lrr = 0.001
epo = 100
for op_idx,opti in enumerate(opti_list):
    for ac_idx,acti in enumerate(acti_list):
        modelpath = f'./data/hdf5/8_deltime_data_plus_conv1d_{op_idx}_{ac_idx}.hdf5'
        model = build_model(acti, opti, lrr)
        # model = load_model(modelpath, custom_objects={'leaky_relu':tf.nn.leaky_relu, 'mish':mish})

        # 훈련
        er,mo,lr = callbacks(modelpath) 
        # history = model.fit(x_train, y_train, verbose=1, batch_size=batch, epochs = epo, validation_data=(x_val,y_val), callbacks = [er, lr, mo])
        # history_list.append(history)

        # 학습 완료된 모델 저장
        if os.path.exists(modelpath):
            # 기존에 학습된 모델 불러들이기
            model = load_model(modelpath, custom_objects={'leaky_relu':tf.nn.leaky_relu, 'mish':mish})
        else:
            # 학습한 모델이 없으면 파일로 저장
            history = model.fit(x_train, y_train, verbose=1, batch_size=batch, epochs = epo, validation_data=(x_val,y_val), callbacks = [er, lr, mo])

        score, y_predict = evaluate_list(model)
        # 엑셀 추가 코드 
        # 경로 변경 필요!!!!
        df = pd.DataFrame(y_pred)
        df[f'{op_idx}_{ac_idx}'] = y_predict
        df.to_csv(f'./data/csv/8_deltime_data_plus_conv1d_{op_idx}_{ac_idx}.csv',index=False)

        fold_score_list.append(score)
        print('=============parameter=================')
        print('optimizer : ', opti, '\n activation : ', acti, '\n batch_size : ', batch, '\n lr : ', lrr, '\n epochs : ', epo)
        print('r2   : ', score[0])
        print('rmse : ', score[1])
        print('mae : ', score[2])
        print('mse : ', score[3])
        print(f'======================================')


terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time))

print('=========================final score========================')
print("r2           rmse          mae            mse: ")
fold_score_list = np.array(fold_score_list).reshape(len(opti_list),len(acti_list),4)
print(fold_score_list)

# ======================================
# 7939.901074초 걸렸습니다.
# =========================final score========================
# r2           rmse          mae            mse:
# [[[ 8.53577381e-01  1.48523832e+00  4.39713014e-01  2.20593288e+00]
#   [-1.12509477e+02  4.13531049e+01  7.75462672e+00  1.71007929e+03]
#   [ 8.54771203e-01  1.47917116e+00  4.45156667e-01  2.18794732e+00]
#   [ 8.67629098e-01  1.41217437e+00  3.89739468e-01  1.99423645e+00]
#   [ 8.48643198e-01  1.51005593e+00  4.31390688e-01  2.28026891e+00]
#   [ 8.65744532e-01  1.42219141e+00  4.14320103e-01  2.02262841e+00]
#   [ 8.45768571e-01  1.52432826e+00  4.34890243e-01  2.32357666e+00]]

#  [[ 8.43679337e-01  1.53461788e+00  4.76519622e-01  2.35505205e+00]
#   [ 6.95606495e-01  2.14145894e+00  9.97728565e-01  4.58584640e+00]
#   [-2.00136348e-03  3.88531485e+00  1.36251516e+00  1.50956714e+01]
#   [ 8.41683716e-01  1.54438244e+00  4.68517620e-01  2.38511712e+00]
#   [ 8.28216610e-01  1.60872809e+00  5.03426168e-01  2.58800607e+00]
#   [ 8.44333467e-01  1.53140368e+00  4.81995056e-01  2.34519724e+00]
#   [ 8.60634306e-01  1.44900539e+00  3.92372044e-01  2.09961663e+00]]

#  [[ 8.57726119e-01  1.46404576e+00  4.29788004e-01  2.14342999e+00]
#   [ 7.61695496e-01  1.89477736e+00  7.79693224e-01  3.59018124e+00]
#   [ 8.51596449e-01  1.49525137e+00  4.08613711e-01  2.23577665e+00]
#   [ 8.46038107e-01  1.52299572e+00  4.72011879e-01  2.31951595e+00]
#   [ 7.85436889e-01  1.79791680e+00  5.39434762e-01  3.23250481e+00]
#   [ 8.46641907e-01  1.52000638e+00  4.57490261e-01  2.31041940e+00]
#   [ 8.60269401e-01  1.45090114e+00  3.87215092e-01  2.10511411e+00]]

#  [[ 1.49328199e-01  3.57991800e+00  1.40456869e+00  1.28158129e+01]
#   [ 1.85189193e-01  3.50364787e+00  1.42275688e+00  1.22755484e+01]
#   [ 2.03909274e-01  3.46316628e+00  1.46704769e+00  1.19935207e+01]
#   [ 1.50090143e-01  3.57831439e+00  1.41220233e+00  1.28043338e+01]
#   [ 2.74568873e-01  3.30590337e+00  1.01414845e+00  1.09289971e+01]
#   [ 1.82198986e-01  3.51007086e+00  1.43824544e+00  1.23205974e+01]
#   [ 2.35050661e-01  3.39475470e+00  7.88102336e-01  1.15243595e+01]]

#  [[ 8.19342307e-01  1.64975818e+00  5.17919626e-01  2.72170206e+00]
#   [-1.42880533e+01  1.51763787e+01  5.86796283e+00  2.30322470e+02]
#   [ 7.66186988e-01  1.87683632e+00  9.90009406e-01  3.52251458e+00]
#   [ 8.59328947e-01  1.45577558e+00  4.39334726e-01  2.11928255e+00]
#   [ 8.45715532e-01  1.52459035e+00  4.57265025e-01  2.32437572e+00]
#   [ 8.51832075e-01  1.49406386e+00  4.59309919e-01  2.23222682e+00]
#   [ 8.47146108e-01  1.51750563e+00  4.42634435e-01  2.30282334e+00]]

#  [[ 2.71967792e-01  3.31182483e+00  1.26655125e+00  1.09681837e+01]
#   [ 4.71096036e-01  2.82280236e+00  9.66387398e-01  7.96821318e+00]
#   [ 4.28746055e-01  2.93363898e+00  1.03915164e+00  8.60623765e+00]
#   [ 5.01368833e-01  2.74082793e+00  9.97341283e-01  7.51213775e+00]
#   [ 3.64235293e-01  3.09485473e+00  1.14562659e+00  9.57812582e+00]
#   [ 4.89754937e-01  2.77256328e+00  9.03994404e-01  7.68710713e+00]
#   [ 3.92672602e-01  3.02484760e+00  6.17421627e-01  9.14970298e+00]]]