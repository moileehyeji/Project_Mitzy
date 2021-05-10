import numpy as np
import db_connect as db
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import  LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import all_estimators




# db 직접 불러오기 

# query = "SELECT a.date, IF(DATE LIKE '2019-%', '2019', '2020') AS YEAR , CASE WHEN DATE LIKE '%-01-%' THEN '1' WHEN  DATE LIKE '%-02-%' THEN '2' WHEN  DATE LIKE '%-03-%' THEN '3' WHEN  DATE LIKE '%-04-%' THEN '4'\
# WHEN  DATE LIKE '%-05-%' THEN '5' WHEN  DATE LIKE '%-06-%' THEN '6' WHEN  DATE LIKE '%-07-%' THEN '7' WHEN  DATE LIKE '%-08-%' THEN '8' WHEN  DATE LIKE '%-09-%' THEN '9' WHEN  DATE LIKE '%-10-%' THEN '10' WHEN  DATE LIKE '%-11-%' THEN '11' ELSE '12' END AS MONTH,\
# DAYOFWEEK (DATE)AS DAY, a.time, s.index AS category, d.index AS dong, a.value FROM `business_location_data` AS a INNER JOIN `category_table` AS s ON a.category = s.category INNER JOIN `location_table` AS d ON a.dong = d.location WHERE si = '서울특별시'"
query = "SELECT * FROM `main_data_table`"

db.cur.execute(query)
dataset = np.array(db.cur.fetchall())


# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)

db.connect.commit()


# train, test 나누기

train_value = df[ '2020-09-01' >= df['date'] ]

x_train = train_value.iloc[:,1:-1]
y_train = train_value.iloc[:,-1]

test_value = df[df['date'] >=  '2020-09-01']

x_test = test_value.iloc[:,1:-1]
y_test = test_value.iloc[:,-1]

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


# ARDRegression 의 정답률 :  [0.09229221 0.09205571 0.09104287 0.09263682 0.09190473]
# AdaBoostRegressor 의 정답률 :  [0.41024962 0.44708276 0.4592201  0.31230852 0.41216578]
# BaggingRegressor 의 정답률 :  [0.81020433 0.80685568 0.79667401 0.80632686 0.80283089]*************
# BayesianRidge 의 정답률 :  [0.08988594 0.09277902 0.09354932 0.09024272 0.09358653]
# CCA 의 정답률 :  [-0.36394461 -0.3629555  -0.34884139 -0.3339627  -0.34438774]
# DecisionTreeRegressor 의 정답률 :  [0.79499496 0.79491859 0.79672621 0.78625928 0.78846612]*******
# DummyRegressor 의 정답률 :  [-1.39460248e-04 -2.71865678e-07 -7.23742342e-05 -2.69246494e-06
#  -2.02652557e-06]
# ElasticNet 의 정답률 :  [0.08932776 0.09149547 0.09294653 0.08813294 0.08978014]
# ElasticNetCV 의 정답률 :  [0.09255558 0.09158837 0.09065064 0.09295939 0.09207772]
# ExtraTreeRegressor 의 정답률 :  [0.79684475 0.78080015 0.79591355 0.78890953 0.7949307 ]***********
# ExtraTreesRegressor 의 정답률 :  [0.79798228 0.78623229 0.79390681 0.79277296 0.80143356]***********
# GammaRegressor 의 정답률 :  [nan nan nan nan nan]
# GaussianProcessRegressor 의 정답률 :  [nan nan nan nan nan]
# GradientBoostingRegressor 의 정답률 :  [0.59267656 0.58675527 0.5841978  0.58931798 0.5835008 ]
# HistGradientBoostingRegressor 의 정답률 :  [0.80679224 0.81036163 0.81072782 0.81588779 0.80795238]**************
# HuberRegressor 의 정답률 :  [ 0.00157612  0.00060046  0.00500124 -0.00086127  0.00322836]
# IsotonicRegression 의 정답률 :  [nan nan nan nan nan]
# KNeighborsRegressor 의 정답률 :  [nan nan nan nan nan]
# KernelRidge 의 정답률 :  [nan nan nan nan nan]
# Lars 의 정답률 :  [0.09046475 0.09187264 0.09223953 0.09209678 0.09345632]
# LarsCV 의 정답률 :  [0.09052237 0.09162817 0.0906914  0.09187588 0.09271878]
# Lasso 의 정답률 :  [0.08823851 0.08932343 0.09017958 0.09063951 0.08925399]
# LassoCV 의 정답률 :  [0.09289299 0.09203842 0.09427049 0.08926573 0.09152858]
# LassoLars 의 정답률 :  [-1.18110650e-05 -1.89500133e-05 -2.90888067e-05 -1.50586039e-05
#  -4.38230082e-05]
# LassoLarsCV 의 정답률 :  [0.09148023 0.09047832 0.08964153 0.09300294 0.09268461]
# LassoLarsIC 의 정답률 :  [0.09188294 0.09236896 0.09230015 0.09036534 0.09305611]
# LinearRegression 의 정답률 :  [0.09353405 0.09122129 0.09103626 0.08996892 0.09398158]
# LinearSVR 의 정답률 :  [-0.07772151 -0.08099265 -0.05306037 -0.02736828  0.07609762]
# MLPRegressor 의 정답률 :  [ 0.20004422  0.09191791  0.22989137 -0.01874852  0.19783834]
# MultiOutputRegressor 은 없는 놈!
# MultiTaskElasticNet 의 정답률 :  [nan nan nan nan nan]
# MultiTaskElasticNetCV 의 정답률 :  [nan nan nan nan nan]
# MultiTaskLasso 의 정답률 :  [nan nan nan nan nan]
# MultiTaskLassoCV 의 정답률 :  [nan nan nan nan nan]
# NuSVR 의 정답률 :  [-0.01705061 -0.01682572 -0.01796076 -0.01931741 -0.01783602]
# OrthogonalMatchingPursuit 의 정답률 :  [0.04889297 0.0480308  0.05014399 0.05089293 0.0509427 ]
# OrthogonalMatchingPursuitCV 의 정답률 :  [0.08982726 0.09066331 0.0928185  0.09274297 0.09281907]
# PLSCanonical 의 정답률 :  [-0.40781833 -0.36228051 -0.35945378 -0.4001784  -0.38946564]
# PLSRegression 의 정답률 :  [0.0898516  0.09193652 0.09352827 0.09240929 0.0906477 ]
# PassiveAggressiveRegressor 의 정답률 :  [-0.38030772  0.08447564  0.01725782  0.01424516 -0.01654324]
# PoissonRegressor 의 정답률 :  [nan nan nan nan nan]
# RANSACRegressor 의 정답률 :  [-0.23032443 -0.17226278 -0.16941531 -0.13107626 -0.22263323]
# RadiusNeighborsRegressor 의 정답률 :  [nan nan nan nan nan]
# RandomForestRegressor 의 정답률 :  [0.81647815 0.80939832 0.81048145 0.8082622  0.8080544 ]************
# RegressorChain 은 없는 놈!
# Ridge 의 정답률 :  [0.0930643  0.09140443 0.09387827 0.08997452 0.09172452]
# RidgeCV 의 정답률 :  [0.09273615 0.09344641 0.09103492 0.09012986 0.09242517]
# SGDRegressor 의 정답률 :  [-2.70159448e+27 -8.29659859e+26 -6.46404433e+26 -1.51964775e+27
#  -2.00996900e+27]