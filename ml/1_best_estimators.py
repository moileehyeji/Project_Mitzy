import numpy as np
import db_connect as db
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils import all_estimators




# db 직접 불러오기 

# 0 없다
'''
query = "SELECT a.date, IF(DATE LIKE '2019-%', '2019', '2020') AS YEAR , CASE WHEN DATE LIKE '%-01-%' THEN '1' WHEN  DATE LIKE '%-02-%' THEN '2' WHEN  DATE LIKE '%-03-%' THEN '3' WHEN  DATE LIKE '%-04-%' THEN '4'\
WHEN  DATE LIKE '%-05-%' THEN '5' WHEN  DATE LIKE '%-06-%' THEN '6' WHEN  DATE LIKE '%-07-%' THEN '7' WHEN  DATE LIKE '%-08-%' THEN '8' WHEN  DATE LIKE '%-09-%' THEN '9' WHEN  DATE LIKE '%-10-%' THEN '10' WHEN  DATE LIKE '%-11-%' THEN '11' ELSE '12' END AS MONTH,\
DAYOFWEEK (DATE)AS DAY, a.time, s.index AS category, d.index AS dong, a.value FROM `business_location_data` AS a INNER JOIN `category_table` AS s ON a.category = s.category INNER JOIN `location_table` AS d ON a.dong = d.location WHERE si = '서울특별시'"
'''

# 0 있다
query = "select * from main_data_table"


db.cur.execute(query)
dataset = np.array(db.cur.fetchall())


# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)

db.connect.commit()


# train, test 나누기

train_value = df[ '2020-09-01' >= df['date'] ]

x_train = train_value.iloc[:,1:-1].astype('int64').to_numpy()
y_train = train_value.iloc[:,-1].astype('int64').to_numpy()

test_value = df[df['date'] >=  '2020-09-01']

x_pred = test_value.iloc[:,1:-1].astype('int64').to_numpy()
y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()

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


# ARDRegression 의 정답률 :  [0.06604899 0.06663295 0.06734765 0.06719854 0.0658527 ]
# AdaBoostRegressor 의 정답률 :  [ 0.31341365 -0.19947769  0.28921458  0.26070735  0.31974546]
# BaggingRegressor 의 정답률 :  [0.8310621  0.83330123 0.83586496 0.83082827 0.83515124]******1
# BayesianRidge 의 정답률 :  [0.06639984 0.06651518 0.06752203 0.06566651 0.06705282]
# CCA 의 정답률 :  [-0.48836523 -0.46416926 -0.47042925 -0.49745129 -0.4768754 ]
# DecisionTreeRegressor 의 정답률 :  [0.82099855 0.82805213 0.82044281 0.81491943 0.82689659]******3
# DummyRegressor 의 정답률 :  [-1.27781643e-06 -3.90698776e-08 -1.12121311e-07 -2.86487409e-07
#  -5.72367544e-07]
# ElasticNet 의 정답률 :  [0.0641608  0.06438003 0.06527759 0.06256866 0.06522079]
# ElasticNetCV 의 정답률 :  [0.06743163 0.06708649 0.06667871 0.06540715 0.06641723]
# ExtraTreeRegressor 의 정답률 :  [0.81953081 0.8224338  0.82904534 0.83048541 0.82785028]******2
# ExtraTreesRegressor 의 정답률 :  [0.818857   0.82392771 0.82447131 0.82767775 0.82730721]******4
# GammaRegressor 의 정답률 :  [nan nan nan nan nan]
# GaussianProcessRegressor 의 정답률 :  [nan nan nan nan nan]
# GradientBoostingRegressor 의 정답률 :  [0.52399317 0.52991136 0.53434446 0.53684388 0.5322593 ]
# HistGradientBoostingRegressor 의 정답률 :  [0.81466868 0.8090357  0.82109766 0.81032493 0.80565226]******5
# HuberRegressor 의 정답률 :  [-0.0299433  -0.03025516 -0.02999328 -0.02995172 -0.0292951 ]
# IsotonicRegression 의 정답률 :  [nan nan nan nan nan]
# KNeighborsRegressor 의 정답률 :  [0.79068543 0.79262399 0.79379278 0.7899116  0.79794595]******6
# KernelRidge 의 정답률 :  [nan nan nan nan nan]
# Lars 의 정답률 :  [0.06655326 0.06542301 0.06631942 0.06774537 0.06706809]
# LarsCV 의 정답률 :  [0.06663651 0.06733573 0.06566837 0.06740469 0.06598947]
# Lasso 의 정답률 :  [0.05931243 0.05837468 0.05896962 0.05978989 0.05827342]
# LassoCV 의 정답률 :  [0.06563901 0.0673635  0.06766276 0.06621268 0.06615096]
# LassoLars 의 정답률 :  [-4.81357799e-06 -1.25702235e-06 -5.69212051e-08 -9.74625234e-07
#  -9.65563822e-08]
# LassoLarsCV 의 정답률 :  [0.06725823 0.06705479 0.06644651 0.06584984 0.06640511]
# LassoLarsIC 의 정답률 :  [0.06674525 0.06659274 0.06720356 0.06638796 0.06618187]
# LinearRegression 의 정답률 :  [0.06726119 0.06535984 0.06642777 0.06749403 0.06659119]