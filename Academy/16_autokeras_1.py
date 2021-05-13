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
import autokeras as ak

# db 직접 불러오기 
fold_score_list = []
history_list=[]


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
    x = Conv1D(filters=1024,kernel_size=2,padding='same',strides = 2, activation=acti)(inputs)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(256, activation=acti)(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation=acti)(x)
    x = Dense(16, activation=acti)(x)
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

print(x_train.shape, y_train.shape) #(3472128, 42) (3472128,)
print(x_pred.shape, y_pred.shape)   #(177408, 42) (177408,) 

# model = ak.StructuredDataRegressor(
#                             overwrite=True, 
#                             max_trials=1,     
#                             loss = 'mse',
#                             metrics=['mae'],
#                             project_name='16_autokeras',
#                             directory='./mitzy/data/structured_data_regressor'  
# )
# 데이터 형식: 2차원

# model.fit(x_train, y_train, epochs=10, validation_split=0.2)#, callbacks = [es, re, mo]) # validation_splitr 기본 0.2

model = load_model('./mitzy/data/structured_data_regressor/16_autokeras/best_model')

model.summary()

results, y_predict = evaluate_list(model)
print(results) 

#=====================================
import matplotlib.pyplot as plt
# summarize history for accuracy
fig = plt.figure(figsize = (12,4))
chart = fig.add_subplot(1,1,1)

chart.plot(y_predict, marker = 'o', color = 'blue', label = 'pred')
chart.plot(y_pred, marker = '^', color = 'red', label = 'actual')
chart.set_title('autokeras predict', size = 30)
plt.xlabel('date')
plt.ylabel('delivery')
plt.legend(loc='best')
plt.show()
#=====================================

###16_autokeras
# Search: Running Trial #1

# Hyperparameter    |Value             |Best Value So Far
# structured_data...|True              |?
# structured_data...|2                 |?
# structured_data...|False             |?
# structured_data...|0                 |?
# structured_data...|32                |?
# structured_data...|32                |?
# regression_head...|0                 |?
# optimizer         |adam              |?
# learning_rate     |0.001             |?
# r2 :  1.0
# rmse :  3.21227811755653e-36
# mae :  3.138422771184618e-36
# mse :  1.0318730704532525e-71
# [1.0, 3.21227811755653e-36, 3.138422771184618e-36, 1.0318730704532525e-71]
'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 42)]              0
_________________________________________________________________
multi_category_encoding (Mul (None, 42)                0
_________________________________________________________________
normalization (Normalization (None, 42)                85
_________________________________________________________________
dense (Dense)                (None, 32)                1376
_________________________________________________________________
re_lu (ReLU)                 (None, 32)                0
_________________________________________________________________
dense_1 (Dense)              (None, 32)                1056
_________________________________________________________________
re_lu_1 (ReLU)               (None, 32)                0
_________________________________________________________________
regression_head_1 (Dense)    (None, 1)                 33
=================================================================
Total params: 2,550
Trainable params: 2,465
Non-trainable params: 85
_________________________________________________________________
'''
