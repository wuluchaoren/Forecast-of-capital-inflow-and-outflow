import pandas as pd
import numpy  as np
import math
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm
import warnings

pd.options.mode.chained_assignment = None

# 读取数据文件
# 用户数据:id 性别 城市 星座
user_profile_table = pd.read_csv(r"../data/user_profile_table.csv", sep=',', engine='python', encoding='utf-8')
# 用户申购赎回数数据
user_balance_table = pd.read_csv(r"../data/user_balance_table.csv", sep=',', engine='python', encoding='utf-8',
                                 parse_dates=['report_date'])

# user_balance_table 数据缺失值处理 填充0(众数)
user_balance_table = user_balance_table.fillna(0)

# 购买赎回总量计算
user_balance = user_balance_table.groupby(['report_date'])
purchase_redeem_total = user_balance['total_purchase_amt', 'total_redeem_amt'].sum()

prtwd = purchase_redeem_total
prtwd = prtwd.fillna(method='pad')
# 选出2014年 3月份到7月的数据 的数据
# prtwd1307 = prtwd['2013-07']
# prtwd1308 = prtwd['2013-08']
# prtwd1309 = prtwd['2013-09']
# prtwd1310 = prtwd['2013-10']
# prtwd1311 = prtwd['2013-11']
# prtwd1312 = prtwd['2013-12']
# prtwd1401 = prtwd['2014-01']
# prtwd1402 = prtwd['2014-02']
prtwd1403 = prtwd['2014-03']
prtwd1404 = prtwd['2014-04']
prtwd1405 = prtwd['2014-05']
prtwd1406 = prtwd['2014-06']
prtwd1407 = prtwd['2014-07']
prtwd1408 = prtwd['2014-08']
# 画3-7每月ts图
# fig1307 = plt.figure(figsize=(12, 8))
# prtwd1307.plot()
# plt.title('1307 ts')
# fig1308 = plt.figure(figsize=(12, 8))
# prtwd1308.plot()
# plt.title('1308 ts')
# fig1309 = plt.figure(figsize=(12, 8))
# prtwd1309.plot()
# plt.title('1309 ts')
# fig1310 = plt.figure(figsize=(12, 8))
# prtwd1310.plot()
# plt.title('1310 ts')
# fig1311 = plt.figure(figsize=(12, 8))
# prtwd1311.plot()
# plt.title('1311 ts')
# fig1312 = plt.figure(figsize=(12, 8))
# prtwd1312.plot()
# plt.title('1312 ts')
# fig1401 = plt.figure(figsize=(12, 8))
# prtwd1401.plot()
# plt.title('1401 ts')
# fig1402 = plt.figure(figsize=(12, 8))
# prtwd1402.plot()
# plt.title('1402 ts')
fig1403 = plt.figure(figsize=(12, 8))
prtwd1403.plot()
plt.title('1403 ts')
fig1404 = plt.figure(figsize=(12, 8))
prtwd1404.plot()
plt.title('1404 ts')
fig1405 = plt.figure(figsize=(12, 8))
prtwd1405.plot()
plt.title('1405 ts')
fig1406 = plt.figure(figsize=(12, 8))
prtwd1406.plot()
plt.title('1406 ts')
fig1407 = plt.figure(figsize=(12, 8))
prtwd1407.plot()
plt.title('1407 ts')
plt.show()
