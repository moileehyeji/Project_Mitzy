import numpy as np
import db_connect as db
import pandas as pd
import timeit
import tensorflow as tf
import numpy as np

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
    
    inputs = Input(shape=(x_train.shape[1],1,1),name='input')

    x = Conv2D(filters=32,kernel_size=(2,2),padding='same',strides = (2,2), activation=acti)(inputs)
    x = BatchNormalization()(x)
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

x_pred, y_pred = load_data("select * from main_data_table ORDER BY DATE, YEAR, MONTH ,TIME, category ASC")
x_train, y_train = load_data("SELECT * FROM main_data_table WHERE VALUE != 0 ")

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1,1)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1],1,1)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  train_size=0.9, random_state = 77, shuffle=True ) 

print(x_train.shape, y_train.shape) #(2213481, 46, 1) (2213481,)
print(x_pred.shape, y_pred.shape)   #(177408, 46, 1) (177408,)
print(x_val.shape, y_val.shape)     #(245943, 46, 1) (245943,) 

leaky_relu = tf.nn.leaky_relu
acti_list = [leaky_relu, mish, 'swish', 'elu', 'relu', 'selu','tanh']
opti_list = [RMSprop, Nadam, Adam, Adadelta, Adamax, Adagrad]
batch = 1024
lrr = 0.0001
epo = 200
for op_idx,opti in enumerate(opti_list):
    for ac_idx,acti in enumerate(acti_list):
        modelpath = f'./data/hdf5/1_0del_conv2d_{op_idx}_{ac_idx}.hdf5'
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
        df.to_csv(f'./data/csv/1_0del_conv2d_{op_idx}_{ac_idx}.csv',index=False)

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


''' 
============================================================
17146.344947초 걸렸습니다.
=========================final score========================
r2           rmse          mae            mse:
[[[-4.93243043e-01  4.29459858e+00  2.81921315e+00  1.84435770e+01]
  [-2.79027708e+00  6.84215339e+00  3.56794375e+00  4.68150630e+01]
  [-6.55314589e+00  9.65875751e+00  5.36031877e+00  9.32915967e+01]
  [-5.03807706e-01  4.30976390e+00  2.97666530e+00  1.85740649e+01]
  [-7.01916687e-01  4.58486418e+00  3.32087720e+00  2.10209795e+01]
  [-6.47052277e-01  4.51035804e+00  3.12437153e+00  2.03433297e+01]
  [-7.44153902e+00  1.02109969e+01  5.55600597e+00  1.04264457e+02]]

 [[-1.14012803e-01  3.70938932e+00  2.63531322e+00  1.37595692e+01]
  [-4.99027170e-01  4.30290818e+00  2.86418204e+00  1.85150188e+01]
  [-3.07189994e-01  4.01815498e+00  2.62666688e+00  1.61455695e+01]
  [-4.82072498e-02  3.59816360e+00  2.44919247e+00  1.29467813e+01]
  [-1.64444533e-01  3.79242263e+00  2.82570938e+00  1.43824694e+01]
  [-1.25710192e-02  3.53647077e+00  2.41894893e+00  1.25066255e+01]
  [-1.05353734e+00  5.03626564e+00  2.95806819e+00  2.53639716e+01]]

 [[-1.60979049e-01  3.78677514e+00  2.56635414e+00  1.43396660e+01]
  [-4.00121539e-02  3.58407042e+00  2.42862500e+00  1.28455608e+01]
  [-1.31743058e+00  5.35008514e+00  3.45064685e+00  2.86234110e+01]
  [-8.48360544e-02  3.66049132e+00  2.52519814e+00  1.33991967e+01]
  [-1.60313709e-01  3.78568992e+00  2.82720174e+00  1.43314481e+01]
  [-2.64786319e-01  3.95244564e+00  2.82997243e+00  1.56218266e+01]
  [-1.75500995e+00  5.83336179e+00  3.25523323e+00  3.40281098e+01]]

 [[-2.10660329e+00  6.19441397e+00  5.47345640e+00  3.83707644e+01]
  [-6.28228005e-01  4.48450938e+00  3.50613104e+00  2.01108244e+01]
  [-7.27773461e-01  4.61956121e+00  3.34816681e+00  2.13403458e+01]
  [-6.15308687e-01  4.46668258e+00  3.51649389e+00  1.99512533e+01]
  [-1.26781444e+00  5.29250266e+00  3.97057820e+00  2.80105844e+01]
  [-7.87550565e-01  4.69879496e+00  3.41423071e+00  2.20786741e+01]
  [-2.18958805e+00  6.27660248e+00  5.76280616e+00  3.93957387e+01]]

 [[-3.36771803e-01  4.06336621e+00  2.72356681e+00  1.65109449e+01]
  [-2.03181726e-01  3.85498720e+00  2.58232003e+00  1.48609263e+01]
  [-4.21644328e-01  4.19037418e+00  2.89846245e+00  1.75592357e+01]
  [-1.33522877e-01  3.74173022e+00  2.64934560e+00  1.40005450e+01]
  [-1.57062122e-01  3.78038181e+00  2.75753440e+00  1.42912866e+01]
  [-5.18867258e-02  3.60447331e+00  2.70815457e+00  1.29922278e+01]
  [-4.06289565e-01  4.16768324e+00  2.81298422e+00  1.73695836e+01]]

 [[-1.91477427e+00  6.00011799e+00  4.98733454e+00  3.60014158e+01]
  [-3.80675307e-01  4.12955358e+00  3.03613720e+00  1.70532128e+01]
  [-3.03940767e-01  4.01315800e+00  2.88796419e+00  1.61054371e+01]
  [-1.88824351e-01  3.83191770e+00  2.87003609e+00  1.46835932e+01]
  [-5.31076762e-01  4.34866356e+00  3.29700711e+00  1.89108747e+01]
  [-4.80000989e-01  4.27551397e+00  3.34977586e+00  1.82800197e+01]
  [-6.47873106e-01  4.51148180e+00  3.36855781e+00  2.03534680e+01]]]
'''