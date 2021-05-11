import numpy as np
import db_connect as db
import pandas as pd
import warnings
import joblib
import timeit
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_absolute_error as mae_
from sklearn.metrics import mean_squared_error as mse_
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
from tensorflow.keras.activations import tanh, relu, elu, selu, swish
from tensorflow.keras.layers import LeakyReLU, PReLU
from time import time


# db 직접 불러오기 =================================================

# 0 없다
# query = "SELECT a.date, IF(DATE LIKE '2019-%', '2019', '2020') AS YEAR , CASE WHEN DATE LIKE '%-01-%' THEN '1' WHEN  DATE LIKE '%-02-%' THEN '2' WHEN  DATE LIKE '%-03-%' THEN '3' WHEN  DATE LIKE '%-04-%' THEN '4'\
# WHEN  DATE LIKE '%-05-%' THEN '5' WHEN  DATE LIKE '%-06-%' THEN '6' WHEN  DATE LIKE '%-07-%' THEN '7' WHEN  DATE LIKE '%-08-%' THEN '8' WHEN  DATE LIKE '%-09-%' THEN '9' WHEN  DATE LIKE '%-10-%' THEN '10' WHEN  DATE LIKE '%-11-%' THEN '11' ELSE '12' END AS MONTH,\
# DAYOFWEEK (DATE)AS DAY, a.time, s.index AS category, d.index AS dong, a.value FROM `business_location_data` AS a INNER JOIN `category_table` AS s ON a.category = s.category INNER JOIN `location_table` AS d ON a.dong = d.location WHERE si = '서울특별시'"

# 0 있다
query = "SELECT * FROM `main_data_table`"
db.cur.execute(query)
dataset = np.array(db.cur.fetchall())


# pandas 넣기
column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']
df = pd.DataFrame(dataset, columns=column_name)
db.connect.commit()


# train, test 나누기
pred = df.iloc[:,1:-1]
train_value = df[ '2020-09-01' > df['date'] ]
x_train = train_value.iloc[:,1:-1].astype('int64').to_numpy()
y_train = train_value.iloc[:,-1].astype('int64').to_numpy()

test_value = df[df['date'] >=  '2020-09-01']
x_pred = test_value.iloc[:,1:-1].astype('int64').to_numpy()
y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()

# print(x_train1.shape, y_train1.shape) #(3472128, 6) (3472128,)
# print(x_pred.shape, y_pred.shape)   #(177408, 6) (177408,)
#=======================================================================

def callbacks(modelpath):
    er = EarlyStopping(monitor = 'val_loss',patience=30)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
    lr = ReduceLROnPlateau(monitor = 'val_loss',patience=15, factor=0.5)
    return er,mo,lr

def create_hyperparameter() : 
    params = {} #initialize parameters
    params['learning_rate'] = np.random.uniform(0, 0.1)
    params['batch_size'] = np.random.randint(30, 100)
    params['optimizer'] = np.random.choice([RMSprop, Nadam, Adadelta, Adamax, Adagrad, SGD, Adam])
    params['activation'] = np.random.choice (['swish','PReLU','elu','LeakyReLU','tanh', 'relu', 'elu', 'selu'])
    params['kernel_size'] = np.random.randint(2,3)
    params['filters'] = np.random.randint(20, 300)
    params['dropout'] = np.random.uniform(0.2,0.5)

    return params

def build_model():
    params = create_hyperparameter()
    print('=============================params=============================')
    print(params)

    #2. 모델
    inputs = Input(shape = (6,1),name = 'input')
    x = Conv1D(filters=256,kernel_size=params['kernel_size'],padding='same',activation=params['activation'],name='hidden1')(inputs)
    x = Dropout(params['dropout'])(x)
    x = Conv1D(filters=64,kernel_size=params['kernel_size'],padding='same',activation=params['activation'],name='hidden3')(x)
    x = Dropout(params['dropout'])(x)
    outputs = Dense(1,name = 'outputs')(x)
    model = Model(inputs=inputs,outputs=outputs)

    model.compile(loss = 'mse',optimizer = params['optimizer'](learning_rate=params['learning_rate']), metrics = ['mae'])

    #3. 훈련
    modelpath ='C:/Study/mitzy/data/hdf5/8_conv1d_2.hdf5'
    er,mo,lr = callbacks(modelpath) 
    model.fit(x_train, y_train, verbose=1, batch_size=params['batch_size'], epochs = 200, validation_data=(x_valid, y_valid), callbacks = [er, lr, mo])

    return model


kfold = KFold(n_splits=2, shuffle=True)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
start_time = timeit.default_timer()

model = build_model()

finish_time = timeit.default_timer()
time = round(finish_time - start_time, 2)
time_list.append(time)
print('time : ', time)


# 모델로드
model = load_model('C:/Study/mitzy/data/hdf5/8_conv1d_2.hdf5')

#4. 평가, 예측
y_predict = model.predict(x_pred)
print('예측값 : ', y_predict[:5])
print('실제값 : ', y_pred[:5])


# r2_list
r2 = r2_score(y_pred, y_predict)
print('r2 score     :', r2)
# rmse_list
rmse = mse_(y_pred, y_predict, squared=False)
print('rmse score     :', rmse)
# mae_list
mae = mae_(y_pred, y_predict)
print('mae score     :', mae)


# =============================params=============================





