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
    resnet = DenseNet121(include_top=False, input_shape=(42,42,3))
    inputs = Input(shape=(x_train.shape[1],1,1),name='input')
    a = Conv2D(3, kernel_size=(1,1))(inputs)
    a = UpSampling2D(size=(1,42))(a)
    x = resnet(a)
    x = Flatten()(x)
    x = Dense(16, activation=acti)(x)
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

x_train, y_train, x_pred, y_pred = load_data()

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1,1)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1],1,1)

print(x_train.shape, y_train.shape) #(3472128, 42, 1, 1) (3472128,)
print(x_pred.shape, y_pred.shape) #(177408, 42, 1, 1) (177408,)

leaky_relu = tf.nn.leaky_relu
model = build_model(leaky_relu, Adamax, 0.001)

modelpath = f'./mitzy/data/modelcheckpoint/17_transfer_DenseNet121.hdf5'
er,mo,lr = callbacks(modelpath)

model.fit(x_train, y_train, epochs=5, validation_split=0.2, callbacks=[er,mo,lr], batch_size = 120)#, callbacks = [es, re, mo]) # validation_splitr 기본 0.2

results, y_predict = evaluate_list(model)
print(results) 

print('예측값 :', y_predict[15:30])
print('실제값 :',y_pred[15:30])

# r2 :  -3952822.49068128
# rmse :  414.1338000546944
# mae :  412.9980582447795
# mse :  171506.8043477416
# [-3952822.49068128, 414.1338000546944, 412.9980582447795, 171506.8043477416]
# 예측값 : [[407.52594]
#  [404.33972]
#  [413.54868]
#  [413.02173]
#  [373.0041 ]
#  [403.55884]
#  [409.4947 ]
#  [396.70422]
#  [401.9369 ]
#  [453.94424]
#  [413.27362]
#  [417.5232 ]
#  [432.62518]
#  [336.37384]
#  [376.99915]]
# 실제값 : [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]

