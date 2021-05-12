import numpy as np
import db_connect as db
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils import all_estimators

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

kfold = KFold(n_splits=5, shuffle=True)

allAlgorithms = all_estimators(type_filter = 'regressor')

for (name, algorithm) in allAlgorithms:

    try:
        model = algorithm()

        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print(name, '의 정답률 : ', scores)

        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)

    except:
        print(name, '은 없는 놈!')


# ARDRegression 의 정답률 :  [1. 1. 1. 1. 1.]
# AdaBoostRegressor 의 정답률 :  [0.04921081 0.04856494 0.04718064 0.04569788 0.04757161]
# BaggingRegressor 의 정답률 :  [1.         0.99997343 1.         0.99999867 0.99996404]
# BayesianRidge 의 정답률 :  [0.99839394 0.99452622 0.99882426 0.99900853 0.99958965]
# CCA 의 정답률 :  [0.38857634 0.3885641  0.38887282 0.38888108 0.3891821 ]
# DecisionTreeRegressor 의 정답률 :  [1.         1.         1.         1.         0.99996675]
# DummyRegressor 의 정답률 :  [-2.62961054e-06 -7.72669821e-08 -4.77179853e-06 -1.30049426e-06
#  -7.60683417e-07]
# ElasticNet 의 정답률 :  [-4.36255730e-06 -2.79021804e-07 -3.85096044e-09 -1.09347929e-06
#  -1.95671235e-07]
# ElasticNetCV 의 정답률 :  [0.9964263  0.99636852 0.99632657 0.99640859 0.99637982]
# ExtraTreeRegressor 의 정답률 :  [0.99993326 0.99980109 1.         1.         0.99996684]
# ExtraTreesRegressor 의 정답률 :  [0.99999794 0.9999997  0.99999354 0.99999435 0.99999865]
# GammaRegressor 의 정답률 :  [nan nan nan nan nan]
# GaussianProcessRegressor 의 정답률 :  [nan nan nan nan nan]
# GradientBoostingRegressor 의 정답률 :  [0.20355203 0.20694824 0.20982853 0.20648449 0.20811433]
# HistGradientBoostingRegressor 의 정답률 :  [0.99978623 0.99988854 0.9999178  0.99983822 0.99985455]
# HuberRegressor 의 정답률 :  [-0.0477021  -0.04793605 -0.047607   -0.04724259 -0.04761179]
# IsotonicRegression 의 정답률 :  [nan nan nan nan nan]   