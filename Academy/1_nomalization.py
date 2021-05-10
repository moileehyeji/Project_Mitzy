import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pymysql
import math
from scipy.stats import norm
from scipy.stats import zscore 
from scipy.stats import norm
from scipy import stats

matplotlib.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.family'] = "Malgun Gothic" # 한글 변환

def connect_mysql(query):
    connect = pymysql.connect(host='mitzy.c7xaixb8f0ch.ap-northeast-2.rds.amazonaws.com', user='mitzy', password='mitzy1234!', db='mitzy',\
                            charset='utf8')
    cur = connect.cursor()
    query = query
    
    cur.execute(query)
    select = np.array(cur.fetchall())
    connect.commit()

    return select


# def nomalization_graph(y):
#     y = list(map(int, y))

#     # 정규분포
#     plt.figure(figsize=(10,10))
#     sns.distplot(y, rug=True,fit=norm) 
#     plt.title("주문량 분포도",size=15, weight='bold')
#     plt.show()

#     # Q-Q plot & boxplot
#     fig = plt.figure(figsize=(14,14))
#     ax1 = fig.add_subplot(1, 2, 1)
#     ax2 = fig.add_subplot(1, 2, 2)
#     stats.probplot(y, plot=plt) 
#     green_diamond = dict(markerfacecolor='g', marker='D')
#     ax1.boxplot(y, flierprops=green_diamond)
#     plt.show()

def nomalization_graph(y):
    y = list(map(int, y))

    # 정규분포
    sns.distplot(y, rug=True,fit=norm, ax = ax2) 
    ax2.set_title("주문량 분포도")
    # boxplot
    green_diamond = dict(markerfacecolor='g', marker='D')
    ax3.boxplot(y, flierprops=green_diamond)
    ax3.set_title("boxplot")
    #  Q-Q plot
    stats.probplot(y, plot=plt) 
    ax4.set_title("Q-Q plot")
    plt.show()

fig = plt.figure(figsize=(30,10))
# ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(1,3, 1)
ax3 = fig.add_subplot(1,3, 2)
ax4 = fig.add_subplot(1,3, 3)
    
select = connect_mysql("SELECT * FROM `business_location_data` WHERE si = '서울특별시'")
x = select[:,0]
y = select[:,5]
nomalization_graph(y)




    # y값 추이
    # ax1.plot(x, y)
    # ax1.set_title("y값 추이")
# y값 추이
    # y = list(map(int, y))
    # plt.plot(x, y)
    # plt.title('Date')
    # plt.show()
# # 이상치 확인
# def outliers(data_out):
#     quartile_1, q2, quartile_3 = np.percentile(data_out, [25,50,75])
#     print("1사분위 : ", quartile_1)
#     print("q2 : ", q2)
#     print("3사분위 : ", quartile_3)
#     iqr = quartile_3 - quartile_1
#     lower_bound = quartile_1 - (iqr * 1.5)
#     upper_bound = quartile_3 + (iqr * 1.5)
#     return np.where((data_out > upper_bound) | (data_out < lower_bound))
# outlier_loc = outliers(y)
# print("이상치의 위치 : ",outlier_loc)

# # Q-Q plot & boxplot

# fig = plt.figure(figsize=(14,14))
# ax1 = fig.add_subplot(1, 2, 1)
# ax2 = fig.add_subplot(1, 2, 2)
# stats.probplot(y, plot=plt) 
# green_diamond = dict(markerfacecolor='g', marker='D')
# ax1.boxplot(y, flierprops=green_diamond)
# plt.show()