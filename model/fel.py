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
prtwd1403to1407 = prtwd['2013-03':'2014-07']
prtwd1403 = prtwd['2014-03']
prtwd1404 = prtwd['2014-04']
prtwd1405 = prtwd['2014-05']
prtwd1406 = prtwd['2014-06']
prtwd1407 = prtwd['2014-07']
prtwd1408 = prtwd['2014-08']


'''
# 添加节假日
prtwd1403to1407['festival'] = 0
prtwd1403to1407.ix['2014-04-05':'2014-04-07', 'festival'] = 1
prtwd1403to1407.ix['2014-05-01':'2014-05-03', 'festival'] = 1
prtwd1403to1407.ix['2014-05-31':'2014-06-02', 'festival'] = 1
# 添加月初1周
prtwd1403to1407['early_month'] = 0
prtwd1403to1407.ix['2014-03-01':'2014-03-07', 'early_month'] = 1
prtwd1403to1407.ix['2014-04-01':'2014-04-07', 'early_month'] = 1
prtwd1403to1407.ix['2014-05-01':'2014-05-07', 'early_month'] = 1
prtwd1403to1407.ix['2014-06-01':'2014-06-07', 'early_month'] = 1
prtwd1403to1407.ix['2014-07-01':'2014-07-07', 'early_month'] = 1
# 添加月末1周
prtwd1403to1407['late_month'] = 0
prtwd1403to1407.ix['2014-03-25':'2014-03-31', 'late_month'] = 1
prtwd1403to1407.ix['2014-04-24':'2014-04-30', 'late_month'] = 1
prtwd1403to1407.ix['2014-05-25':'2014-05-31', 'late_month'] = 1
prtwd1403to1407.ix['2014-06-24':'2014-06-30', 'late_month'] = 1
prtwd1403to1407.ix['2014-07-25':'2014-07-31', 'late_month'] = 1
# 提取节假日
festival = prtwd1403to1407[prtwd1403to1407['festival'] == 1]
index = pd.date_range('2014-8-21', '2014-8-29')
redeem_festival = pd.DataFrame((festival.loc[:, 'total_redeem_amt']).values, index=index, columns=['total_redeem_amt'])
redeem_festival = redeem_festival.astype('float64')
# 提取月初1周
early_month = prtwd1403to1407[prtwd1403to1407['early_month'] == 1]
index = pd.date_range('2014-6-27', '2014-7-31')
redeem_early = pd.DataFrame((early_month.loc[:, 'total_redeem_amt']).values, index=index, columns=['total_redeem_amt'])
redeem_early = redeem_early.astype('float64')
# 提取月末1周
late_month = prtwd1403to1407[prtwd1403to1407['late_month'] == 1]
index = pd.date_range('2014-7-21', '2014-8-24')
redeem_late = pd.DataFrame((late_month.loc[:, 'total_redeem_amt']).values, index=index, columns=['total_redeem_amt'])
redeem_late = redeem_late.astype('float64')
# 提取全部日期
redeem_data1403to1407 = pd.DataFrame(prtwd1403to1407['total_redeem_amt'])
redeem_data1403to1407 = redeem_data1403to1407.astype('float64')
'''
print('begin ARMA')
'''
warnings.filterwarnings("ignore")
text_file = open("../test/fel/festival.txt", "w")
for p in range(0, 10):
    for q in range(0, 10):
        if p != q:
            try:
                arma_model = sm.tsa.ARMA(redeem_festival, (p, q)).fit()
                text_file.write(
                    'festival redeem ARMA(%d,%d)AIC:%f BIC:%f HQOC:%f' % (p, q, arma_model.aic, arma_model.bic, arma_model.hqic))
                text_file.write('\n')
            except:
                continue

text_file.close()
text_file = open("../test/fel/early.txt", "w")
for p in range(0, 10):
    for q in range(0, 10):
        if p != q:
            try:
                arma_model = sm.tsa.ARMA(redeem_early, (p, q)).fit()
                text_file.write(
                    'early redeem ARMA(%d,%d)AIC:%f BIC:%f HQOC:%f' % (p, q, arma_model.aic, arma_model.bic, arma_model.hqic))
                text_file.write('\n')
            except:
                continue

text_file.close()
text_file = open("../test/fel/late.txt", "w")
for p in range(0, 10):
    for q in range(0, 10):
        if p != q:
            try:
                arma_model = sm.tsa.ARMA(redeem_late, (p, q)).fit()
                text_file.write(
                    'late redeem ARMA(%d,%d)AIC:%f BIC:%f HQOC:%f' % (p, q, arma_model.aic, arma_model.bic, arma_model.hqic))
                text_file.write('\n')
            except:
                continue

text_file.close()
'''
'''
warnings.filterwarnings("ignore")
redeem_festival_model = sm.tsa.ARMA(redeem_festival, (2, 0)).fit()
redeem_early_model = sm.tsa.ARMA(redeem_early, (4, 2)).fit()
redeem_late_model = sm.tsa.ARMA(redeem_late, (6, 1)).fit()
redeem_arma_model = sm.tsa.ARMA(redeem_data1403to1407, (7, 5)).fit()

# 画预测图
redeem_festival_predict = redeem_festival_model.predict('2014-08-30', '2014-08-31', dynamic=False)
redeem_early_predict = redeem_early_model.predict('2014-08-01', '2014-08-07', dynamic=False)
redeem_late_predict = redeem_late_model.predict('2014-08-25', '2014-08-29', dynamic=False)
redeem_other = redeem_arma_model.predict('2014-08-01', '2014-08-31', dynamic=False)
redeem_other = redeem_other['2014-08-08':'2014-08-24']

redeem_predict = ((redeem_early_predict.append(redeem_other)).append(redeem_late_predict)).append(redeem_festival_predict)

fig, ax = plt.subplots(figsize=(12, 8))
redeem_predict.plot(ax=ax, label='predict')
prtwd1408['total_redeem_amt'].plot(ax=ax, label='real')
plt.legend()
plt.title('8\'s redeem predict by divide')
plt.draw()
'''

plt.show()
