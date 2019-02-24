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
# 收益率：日期 万分收益 七日年化收益
mfd_day_share_interest = pd.read_csv(r"../data/mfd_day_share_interest.csv", sep=',', engine='python', encoding='utf-8',
                                     parse_dates=['mfd_date'])
# 银行拆放利率:日期 隔夜利率（%）1周利率（%）2周利率（%）1个月利率（%）3个月利率（%）6个月利率（%）9个月利率（%）
mfd_bank_shibor = pd.read_csv(r"../data/mfd_bank_shibor.csv", sep=',', engine='python', encoding='utf-8',
                              parse_dates=['mfd_date'])

# user_balance_table 数据缺失值处理 填充0(众数)
user_balance_table = user_balance_table.fillna(0)
# 购买赎回总量计算
user_balance = user_balance_table.groupby(['report_date'])
purchase_redeem_total = user_balance['total_purchase_amt', 'total_redeem_amt'].sum()
purchase_redeem_total1403to1407 = purchase_redeem_total['2014-03':'2014-07']
purchase_total = user_balance['total_purchase_amt'].sum()
purchase_total_diff = purchase_total.diff(1)
redeem_total = user_balance['total_redeem_amt'].sum()
redeem_total_diff = redeem_total.diff(1)
'''

# 购买赎回总量计算
user_balance = user_balance_table.groupby(['report_date'])
purchase_redeem_total = user_balance['total_purchase_amt', 'total_redeem_amt'].sum()

# 计算星期列
date = pd.DataFrame(purchase_redeem_total.index)
# date['day_of_week']=date['report_date'].dt.dayofweek
date['day_of_week'] = date['report_date'].dt.weekday_name
tt_date = date.groupby(['report_date'])
tt_date = tt_date['day_of_week'].sum()
# 转化为0 1 哑变量 并进行DF拼接
purchase_redeem_total_with_week_day = pd.concat([pd.get_dummies(tt_date, columns='day_of_week'), purchase_redeem_total],
                                                axis=1)

# 收益
time_mfd_day_share_interest = mfd_day_share_interest.groupby(['mfd_date'])
share_interest = time_mfd_day_share_interest['mfd_daily_yield', 'mfd_7daily_yield'].sum()
# 拆放利率
t_mfd_bank_shibor = (mfd_bank_shibor.groupby(['mfd_date']))
time_mfd_bank_shibor = t_mfd_bank_shibor['Interest_O_N'].sum()
time_mfd_bank_shibor = time_mfd_bank_shibor.fillna(method='pad')

prtwd = pd.concat([purchase_redeem_total_with_week_day, share_interest, time_mfd_bank_shibor, tt_date], axis=1)
prtwd = prtwd.fillna(method='pad')
'''
prtwd = purchase_redeem_total
prtwd = prtwd.fillna(method='pad')

# 选出2014年 3月份到7月的数据 8月的数据
prtwd1403to1407 = prtwd['2014-03':'2014-07']
prtwd1408 = prtwd['2014-08']
'''
# 添加节假日
prtwd1403to1407['holiday_festivel'] = 0
prtwd1403to1407.ix['2014-04-05':'2014-04-07', 'holiday_festivel'] = 1
prtwd1403to1407.ix['2014-05-01':'2014-05-03', 'holiday_festivel'] = 1
prtwd1403to1407.ix['2014-05-31':'2014-06-02', 'holiday_festivel'] = 1

# 添加月初1周 和月末1周
prtwd1403to1407['early_month'] = 0
prtwd1403to1407.ix['2014-03-01':'2014-03-07', 'early_month'] = 1
prtwd1403to1407.ix['2014-04-01':'2014-04-07', 'early_month'] = 1
prtwd1403to1407.ix['2014-05-01':'2014-05-07', 'early_month'] = 1
prtwd1403to1407.ix['2014-06-01':'2014-06-07', 'early_month'] = 1
prtwd1403to1407.ix['2014-07-01':'2014-07-07', 'early_month'] = 1

prtwd1403to1407['late_month'] = 0
prtwd1403to1407.ix['2014-03-25':'2014-03-31', 'late_month'] = 1
prtwd1403to1407.ix['2014-04-24':'2014-04-30', 'late_month'] = 1
prtwd1403to1407.ix['2014-05-25':'2014-05-31', 'late_month'] = 1
prtwd1403to1407.ix['2014-06-24':'2014-06-30', 'late_month'] = 1
prtwd1403to1407.ix['2014-07-25':'2014-07-31', 'late_month'] = 1
'''


# 计算ACF和PACF

purchase_data1403to1407 = pd.DataFrame(prtwd1403to1407['total_purchase_amt'])
purchase_data1403to1407_diff = purchase_data1403to1407  # purchase_data1403to1407.diff(1)
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(purchase_data1403to1407_diff, title='total_purchase_amt acf', lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(purchase_data1403to1407_diff, title='total_purchase_amt pacf', lags=40, ax=ax2)

redeem_data1403to1407 = pd.DataFrame(prtwd1403to1407['total_redeem_amt'])
redeem_data1403to1407_diff = redeem_data1403to1407  # redeem_data1403to1407.diff(1)
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(redeem_data1403to1407_diff, lags=40, title='total_redeem_amt acf', ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(redeem_data1403to1407_diff, lags=40, title='total_redeem_amt pacf', ax=ax2)

redeem_data1403to1407 = redeem_data1403to1407.astype('float64')
purchase_data1403to1407 = purchase_data1403to1407.astype('float64')

print('begin ARMA')

'''
warnings.filterwarnings("ignore")
for p in range(0,15):
    for q in range(0,15):
        if(p!=q):
            try:
                arma_model = sm.tsa.ARMA(redeem_data1403to1407,(p,q)).fit()
                print('redeem ARMA(%d,%d)AIC:%f BIC:%f HQOC:%f'%(p,q,arma_model.aic,arma_model.bic,arma_model.hqic))
            except:
                continue 



for p in range(0,15):
    for q in range(0,8):
        if(p!=q):
            try:            
                arma_model = sm.tsa.ARMA(purchase_data1403to1407,(p,q)).fit()
                print('purchase ARMA(%d,%d)AIC:%f BIC:%f HQOC:%f'%(p,q,arma_model.aic,arma_model.bic,arma_model.hqic))
            except:
                continue
'''

warnings.filterwarnings("ignore")
redeem_arma_model = sm.tsa.ARMA(redeem_data1403to1407, (7, 5)).fit()
purchase_arma_model = sm.tsa.ARMA(purchase_data1403to1407, (6, 4)).fit()

redeem_resid = redeem_arma_model.resid
purchase_resid = purchase_arma_model.resid
# 残差检验
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(redeem_resid.values.squeeze(), lags=40, title='redeem_amt_resid acf', ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(redeem_resid, lags=40, title='redeem_amt_resid pacf', ax=ax2)

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(purchase_resid.values.squeeze(), lags=40, title='purchase_resid acf', ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(purchase_resid, lags=40, title='purchase_resid pacf', ax=ax2)

# D-W检验
print(sm.stats.durbin_watson(redeem_resid.values))
print(sm.stats.durbin_watson(purchase_resid.values))

# 观察是否符合正态分布
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
fig = sm.qqplot(redeem_resid, line='q', ax=ax, fit=True)
plt.title('redeem_resid Q-Q')

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
fig = sm.qqplot(purchase_resid, line='q', ax=ax, fit=True)
plt.title('purchase_resid Q-Q')

redeem_predict = redeem_arma_model.predict('2014-08-01', '2014-08-31', dynamic=False)
fig, ax = plt.subplots(figsize=(12, 8))
redeem_predict.plot(ax=ax, label='predict')
prtwd1408['total_redeem_amt'].plot(ax=ax, label='test')
plt.legend()
plt.title('july\'s total_redeem_amt predict')
plt.draw()
# 计算误差
redeem_error = np.divide(np.abs(np.array(prtwd1408['total_redeem_amt']) - np.array(redeem_predict).T),
                         np.array(prtwd1408['total_redeem_amt']))
print(redeem_error)

purchase_predict = redeem_arma_model.predict('2014-08-01', '2014-08-31', dynamic=False)
fig, ax = plt.subplots(figsize=(12, 8))
purchase_predict.plot(ax=ax, label='predict')
prtwd1408['total_purchase_amt'].plot(ax=ax, label='test')
plt.legend()
plt.title('july\'s total_purchase_amt predict')
plt.draw()
redeem_error = np.divide(np.abs(np.array(prtwd1408['total_purchase_amt']) - np.array(purchase_predict).T),
                         np.array(prtwd1408['total_purchase_amt']))
print(redeem_error)

plt.show()

# 计算误差
# purchase_error=np.divide(np.abs(np.array(Y_test)-np.array(Y_pred).T),Y_test)
# print(purchase_error)



