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
    inputs = Input(shape = (x_train.shape[1],1),name = 'input')
    # 모델2
    x = Conv1D(filters=32,kernel_size=2,padding='same',strides = 2, activation=acti)(inputs)
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
lrr = 0.001
epo = 50
for op_idx,opti in enumerate(opti_list):
    for ac_idx,acti in enumerate(acti_list):
        modelpath = f'./mitzy/data/modelcheckpoint/21_weather_data_conv1d_{op_idx}_{ac_idx}.hdf5'
        model = build_model(acti, opti, lrr)
        # model = load_model(modelpath, custom_objects={'leaky_relu':tf.nn.leaky_relu, 'mish':mish})

        # 훈련
        er,mo,lr = callbacks(modelpath) 
        # history = model.fit(x_train, y_train, verbose=1, batch_size=batch, epochs = epo, validation_data=(x_val,y_val), callbacks = [er, lr, mo])
        # history_list.append(history)

        score, y_predict = evaluate_list(model)
        # 엑셀 추가 코드 
        # 경로 변경 필요!!!!
        df = pd.DataFrame(y_pred)
        df[f'{op_idx}_{ac_idx}'] = y_predict
        df.to_csv(f'./mitzy/data/csv/21_weather_data_conv1d_{op_idx}_{ac_idx}.csv',index=False)

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

# r2           rmse          mae            mse:
# [[[-4.42597158e+01  2.61124711e+01  2.58196582e+01  6.81861148e+02]
#   [-1.37582274e+00  5.98272553e+00  4.56862696e+00  3.57930047e+01]
#   [-3.34831450e+01  2.27926854e+01  2.26175395e+01  5.19506506e+02]
#   [-5.78395011e+00  1.01095863e+01  9.33027266e+00  1.02203735e+02]
#   [-2.54006247e+02  6.19822691e+01  6.18424596e+01  3.84180169e+03]
#   [-1.74846034e+02  5.14704957e+01  5.13239778e+01  2.64921192e+03]
#   [ 7.62458243e-04  3.87995271e+00  1.09844022e+00  1.50540330e+01]]

#  [[-1.96633465e+01  1.76438107e+01  1.74368521e+01  3.11304058e+02]
#   [-9.45596551e+01  3.79427975e+01  3.77424440e+01  1.43965588e+03]
#   [-9.87795062e+01  3.87715119e+01  3.86547574e+01  1.50323013e+03]
#   [-3.73052952e+01  2.40226807e+01  2.37054273e+01  5.77089186e+02]
#   [-5.85421111e+00  1.01618037e+01  9.82141087e+00  1.03262254e+02]
#   [-1.34503455e-01  4.13423322e+00  2.18313109e+00  1.70918843e+01]
#   [-2.09441701e-02  3.92186877e+00  7.89972545e-01  1.53810547e+01]]

#  [[-4.34926363e+02  8.10398499e+01  8.09678639e+01  6.56745728e+03]
#   [-8.06650224e+02  1.10307164e+02  1.10239446e+02  1.21676705e+04]
#   [-2.20674193e+02  5.77895921e+01  5.76574365e+01  3.33963696e+03]
#   [-1.55760162e+01  1.58027308e+01  1.55772831e+01  2.49726301e+02]
#   [-3.64000981e+01  2.37371422e+01  2.35639351e+01  5.63451922e+02]
#   [-9.54392415e+01  3.81170213e+01  3.79212459e+01  1.45290731e+03]
#   [-2.76285876e-02  3.93468663e+00  8.52866120e-01  1.54817589e+01]]

#  [[-4.01548025e+00  8.69257254e+00  8.27779998e+00  7.55608174e+01]
#   [-4.18540232e+01  2.54090169e+01  2.52536957e+01  6.45618138e+02]
#   [-3.14483843e+02  6.89414832e+01  6.88641211e+01  4.75292810e+03]
#   [-1.91220329e+01  1.74111713e+01  1.72083117e+01  3.03148887e+02]
#   [-6.64444035e+02  1.00126222e+02  1.00063545e+02  1.00252603e+04]
#   [-1.75083270e+02  5.15052036e+01  5.13519375e+01  2.65278600e+03]
#   [-2.66227088e-02  3.93276046e+00  7.54296986e-01  1.54666048e+01]]

#  [[-4.13666693e+01  2.52641227e+01  2.49643465e+01  6.38275897e+02]
#   [-1.92735423e+02  5.40252243e+01  5.39355496e+01  2.91872486e+03]
#   [-4.48271924e+01  2.62756632e+01  2.59851739e+01  6.90410478e+02]
#   [-5.91777656e+01  3.01099539e+01  2.98550170e+01  9.06609323e+02]
#   [-1.22446263e+01  1.41257630e+01  1.35832248e+01  1.99537181e+02]
#   [-5.35500684e+00  9.78475762e+00  8.97923607e+00  9.57414818e+01]
#   [-2.09008266e-02  3.92178552e+00  7.75206961e-01  1.53804017e+01]]

#  [[-3.19055637e+01  2.22652066e+01  2.20952329e+01  4.95739423e+02]
#   [-4.92483340e+03  2.72415567e+02  2.72381836e+02  7.42102410e+04]
#   [-5.89574035e+01  3.00547742e+01  2.99173232e+01  9.03289454e+02]
#   [-1.26703073e+03  1.38215564e+02  1.38163191e+02  1.91035421e+04]
#   [-7.01135249e+01  3.27316700e+01  3.25979180e+01  1.07136222e+03]
#   [-1.33605482e+01  1.47088111e+01  1.41736149e+01  2.16349124e+02]
#   [-6.52233171e-03  3.89407013e+00  8.75737308e-01  1.51637822e+01]]]
