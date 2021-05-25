import numpy as np
import db_connect as db
import pandas as pd
import timeit
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils import all_estimators

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

print(x_train.shape, y_train.shape) #(2459424, 42) (2459424,)
print(x_pred.shape, y_pred.shape)   #(177408, 42) (177408,)

kfold = KFold(n_splits=5, shuffle=True)

allAlgorithms = all_estimators(type_filter = 'regressor')

for (name, algorithm) in allAlgorithms:

    try:
        model = algorithm()

        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print(name, '의 정답률 : ', scores)

        # model.fit(x_train, y_train)
        y_predict = model.predict(x_pred)

        # 엑셀 추가 코드 
        # 경로 변경 필요!!!!
        df = pd.DataFrame(y_predict)
        df['test'] = y_predict
        df.to_csv(f'./data/csv/6_deltime_data_best_estimators{name}.csv',index=False)


    except:
        print(name, '은 없는 놈!')


terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time))


# ARDRegression 의 정답률 :  [0.2022104  0.19901219 0.19833081 0.19985322 0.19715216]
# ARDRegression 은 없는 놈!
# AdaBoostRegressor 의 정답률 :  [0.31269926 0.27911601 0.30616791 0.24420117 0.3474517 ]
# AdaBoostRegressor 은 없는 놈!
# BaggingRegressor 의 정답률 :  [0.83412359 0.82885669 0.83536253 0.82573417 0.83523978]
# BaggingRegressor 은 없는 놈!
# BayesianRidge 의 정답률 :  [0.19739156 0.2019378  0.20048413 0.1990117  0.19783923]
# BayesianRidge 은 없는 놈!
# CCA 의 정답률 :  [-1.80517922e-06 -5.61588859e-07 -3.98873913e-06 -1.10109151e-05
#  -1.15098981e-05]
# CCA 은 없는 놈!
# DecisionTreeRegressor 의 정답률 :  [0.82215405 0.81407455 0.82685637 0.82581351 0.82455551]
# DecisionTreeRegressor 은 없는 놈!
# DummyRegressor 의 정답률 :  [-5.71666394e-07 -1.30700688e-06 -1.03432343e-08 -9.78685824e-07
#  -2.33050358e-07]
# DummyRegressor 은 없는 놈!
# ElasticNet 의 정답률 :  [0.00643984 0.00655703 0.00652771 0.00634463 0.00643329]
# ElasticNet 은 없는 놈!
# ElasticNetCV 의 정답률 :  [0.19996873 0.20172778 0.196818   0.19845382 0.19514148]
# ElasticNetCV 은 없는 놈!
# ExtraTreeRegressor 의 정답률 :  [0.82174114 0.82602131 0.82570362 0.82332131 0.82080993]
# ExtraTreeRegressor 은 없는 놈!