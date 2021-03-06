import numpy as np
import db_connect as db
import pandas as pd
import timeit
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten,Dropout,LSTM, Conv2D,Input,Activation, LSTM, Dropout, BatchNormalization
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
    inputs = Input(shape = (x_train.shape[1],1),name = 'input')
    # 모델2
    x = Conv1D(filters=64,kernel_size=2,padding='same',strides = 2, activation=acti)(inputs)
    x = Conv1D(filters=32,kernel_size=2,padding='same',strides = 2, activation=acti)(x)
    x = Flatten()(x)
    x = Dense(32, activation=acti)(x)
    x = BatchNormalization()(x)
    x = Dense(16, activation=acti)(x)
    x = BatchNormalization()(x)
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
    column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong','temperature','rain','wind','humidity', 'value']
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

x_pred, y_pred = load_data("SELECT d.date,YEAR,MONTH, d.day, d.time, category, dong,temperature,rain,wind,humidity,VALUE FROM main_data_table AS d INNER JOIN `weather` AS s ON d.date = s.DATE AND d.time = s.time", is_train = False)
x_train, y_train = load_data("SELECT d.date,YEAR,MONTH, d.day, d.time, category, dong,temperature,rain,wind,humidity,VALUE FROM main_data_table AS d INNER JOIN `weather` AS s ON d.date = s.DATE AND d.time = s.time WHERE (d.TIME != 2 AND d.TIME != 3 AND d.TIME != 4 AND d.TIME != 5  AND d.TIME != 6 AND d.TIME != 7 AND d.TIME != 8) ORDER BY DATE, YEAR, MONTH, DAY, TIME, category, dong ASC")

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
lrr = 0.01
epo = 100
for op_idx,opti in enumerate(opti_list):
    for ac_idx,acti in enumerate(acti_list):
        modelpath = f'./mitzy/data/modelcheckpoint/24_deltime_weather_conv1d_{op_idx}_{ac_idx}.hdf5'
        model = build_model(acti, opti, lrr)
        # model = load_model(modelpath, custom_objects={'leaky_relu':tf.nn.leaky_relu, 'mish':mish})

        # 훈련
        er,mo,lr = callbacks(modelpath) 
        history = model.fit(x_train, y_train, verbose=1, batch_size=batch, epochs = epo, validation_data=(x_val,y_val), callbacks = [er, lr, mo])
        # history_list.append(history)

        score, y_predict = evaluate_list(model)
        # 엑셀 추가 코드 
        # 경로 변경 필요!!!!
        df = pd.DataFrame(y_pred)
        df[f'{op_idx}_{ac_idx}'] = y_predict
        df.to_csv(f'./mitzy/data/csv/24_deltime_weather_conv1d_{op_idx}_{ac_idx}.csv',index=False)

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
# 381.732020초 걸렸습니다.
# =========================final score========================
# r2           rmse          mae            mse:
# [[[-6.52084993e+02  9.91920609e+01  9.91280668e+01  9.83906494e+03]
#   [-5.87251199e+01  2.99964995e+01  2.97372801e+01  8.99789981e+02]
#   [-1.02502321e+02  3.94881789e+01  3.93700139e+01  1.55931627e+03]
#   [-4.88918282e+01  2.74161691e+01  2.72625218e+01  7.51646329e+02]
#   [-2.20756234e+01  1.86452745e+01  1.82399433e+01  3.47646262e+02]
#   [-3.64091987e+01  2.37400300e+01  2.34164986e+01  5.63589026e+02]
#   [-2.88488405e-02  3.93702205e+00  7.47131768e-01  1.55001426e+01]]

#  [[-1.52859920e+02  4.81454014e+01  4.79864147e+01  2.31797968e+03]
#   [-2.33350327e+02  5.94189322e+01  5.92915646e+01  3.53060950e+03]
#   [-1.38532236e+01  1.49589951e+01  1.47288408e+01  2.23771536e+02]
#   [-2.08739739e+02  5.62124382e+01  5.60783386e+01  3.15983821e+03]
#   [-3.08998884e+03  2.15794703e+02  2.15759272e+02  4.65673537e+04]
#   [-3.16872116e+00  7.92489441e+00  6.91833873e+00  6.28039514e+01]
#   [-1.46594280e-01  4.15620487e+00  1.51978413e+00  1.72740389e+01]]

#  [[-3.39305850e+00  8.13533712e+00  7.67025572e+00  6.61837101e+01]
#   [-1.64190171e+02  4.98866294e+01  4.97277242e+01  2.48867580e+03]
#   [-1.43722930e+01  1.52181334e+01  1.49886974e+01  2.31591585e+02]
#   [-3.31522342e+01  2.26830590e+01  2.25047163e+01  5.14521164e+02]
#   [-7.89729791e+00  1.15776690e+01  1.09082601e+01  1.34042418e+02]
#   [-4.51752455e+01  2.63752550e+01  2.62203409e+01  6.95654079e+02]
#   [-4.42640794e-03  3.89001363e+00  9.35075813e-01  1.51322060e+01]]

#  [[-3.27005397e+00  8.02063482e+00  7.55324150e+00  6.43305829e+01]
#   [-1.33617513e+01  1.47094272e+01  1.44764849e+01  2.16367249e+02]
#   [-9.55029204e+00  1.26073643e+01  1.19988525e+01  1.58945634e+02]
#   [-2.21439872e+02  5.78893108e+01  5.77586361e+01  3.35117231e+03]
#   [-1.01728340e+02  3.93402574e+01  3.92221704e+01  1.54765585e+03]
#   [-5.62148556e+02  9.21093142e+01  9.20410631e+01  8.48412576e+03]
#   [-3.11290102e-02  3.94138232e+00  1.71005968e+00  1.55344946e+01]]

#  [[-5.34065922e+01  2.86297677e+01  2.84826086e+01  8.19663596e+02]
#   [-1.85810921e-02  3.91732736e+00  1.71509507e+00  1.53454537e+01]
#   [-8.40400393e+00  1.19027815e+01  1.16192084e+01  1.41676208e+02]
#   [-8.79193015e+00  1.21458025e+01  1.15065298e+01  1.47520518e+02]
#   [-1.50368034e+02  4.77539331e+01  4.76521540e+01  2.28043813e+03]
#   [-6.85594785e+00  1.08790596e+01  1.01618107e+01  1.18353938e+02]
#   [-8.82997761e-02  4.04917299e+00  1.09912202e+00  1.63958019e+01]]

#  [[-2.18691554e+01  1.85616733e+01  1.83600339e+01  3.44535716e+02]
#   [-8.88489749e-01  5.33395536e+00  4.19275120e+00  2.84510798e+01]
#   [-2.46291622e+02  6.10375035e+01  6.09096309e+01  3.72557684e+03]
#   [-2.12176184e+00  6.85791259e+00  5.64385653e+00  4.70309650e+01]
#   [-3.04538878e-01  4.43323318e+00  3.05884318e+00  1.96535564e+01]
#   [-3.63992651e+01  2.37368779e+01  2.34105404e+01  5.63439372e+02]
#   [ 4.97715805e-06  3.88142305e+00  1.15812287e+00  1.50654449e+01]]]