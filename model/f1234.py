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

'''
按照每个月划分为4个部分的模型
'''

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
prtwd1408 = prtwd['2014-08']

# 添加节假日
prtwd1403to1407['festival'] = 0
prtwd1403to1407.ix['2014-04-05':'2014-04-07', 'festival'] = 1
prtwd1403to1407.ix['2014-05-01':'2014-05-03', 'festival'] = 1
prtwd1403to1407.ix['2014-05-31':'2014-06-02', 'festival'] = 1
# 添加第1周 35
prtwd1403to1407['first_month'] = 0
prtwd1403to1407.ix['2014-03-01':'2014-03-07', 'first_month'] = 1
prtwd1403to1407.ix['2014-04-01':'2014-04-07', 'first_month'] = 1
prtwd1403to1407.ix['2014-05-01':'2014-05-07', 'first_month'] = 1
prtwd1403to1407.ix['2014-06-01':'2014-06-07', 'first_month'] = 1
prtwd1403to1407.ix['2014-07-01':'2014-07-07', 'first_month'] = 1
# 添加第2周 45
prtwd1403to1407['second_month'] = 0
prtwd1403to1407.ix['2014-03-08':'2014-03-16', 'second_month'] = 1
prtwd1403to1407.ix['2014-04-08':'2014-04-16', 'second_month'] = 1
prtwd1403to1407.ix['2014-05-08':'2014-05-16', 'second_month'] = 1
prtwd1403to1407.ix['2014-06-08':'2014-06-16', 'second_month'] = 1
prtwd1403to1407.ix['2014-07-08':'2014-07-16', 'second_month'] = 1
# 添加第3周 33
prtwd1403to1407['third_month'] = 0
prtwd1403to1407.ix['2014-03-17':'2014-03-23', 'third_month'] = 1
prtwd1403to1407.ix['2014-04-17':'2014-04-22', 'third_month'] = 1
prtwd1403to1407.ix['2014-05-17':'2014-05-23', 'third_month'] = 1
prtwd1403to1407.ix['2014-06-17':'2014-06-22', 'third_month'] = 1
prtwd1403to1407.ix['2014-07-17':'2014-07-23', 'third_month'] = 1
# 添加第4周 35
prtwd1403to1407['forth_month'] = 0
prtwd1403to1407.ix['2014-03-24':'2014-03-31', 'forth_month'] = 1
prtwd1403to1407.ix['2014-04-23':'2014-04-30', 'forth_month'] = 1
prtwd1403to1407.ix['2014-05-24':'2014-05-31', 'forth_month'] = 1
prtwd1403to1407.ix['2014-06-23':'2014-06-30', 'forth_month'] = 1
prtwd1403to1407.ix['2014-07-24':'2014-07-31', 'forth_month'] = 1
# 提取节假日
festival = prtwd1403to1407[prtwd1403to1407['festival'] == 1]
index = pd.date_range('2014-8-21', '2014-8-29')
redeem_festival = pd.DataFrame((festival.loc[:, 'total_redeem_amt']).values, index=index, columns=['total_redeem_amt'])
redeem_festival = redeem_festival.astype('float64')
# 提取第1周
first_month = prtwd1403to1407[prtwd1403to1407['first_month'] == 1]
index = pd.date_range('2014-6-27', '2014-7-31')
redeem_first = pd.DataFrame((first_month.loc[:, 'total_redeem_amt']).values, index=index, columns=['total_redeem_amt'])
redeem_first = redeem_first.astype('float64')
# 提取第2周
second_month = prtwd1403to1407[prtwd1403to1407['second_month'] == 1]
index = pd.date_range('2014-6-24', '2014-8-07')
redeem_second = pd.DataFrame((second_month.loc[:, 'total_redeem_amt']).values, index=index, columns=['total_redeem_amt'])
redeem_second = redeem_second.astype('float64')
# 提取第3周
third_month = prtwd1403to1407[prtwd1403to1407['third_month'] == 1]
index = pd.date_range('2014-7-15', '2014-8-16')
redeem_third = pd.DataFrame((third_month.loc[:, 'total_redeem_amt']).values, index=index, columns=['total_redeem_amt'])
redeem_third = redeem_third.astype('float64')
# 提取第4周
forth_month = prtwd1403to1407[prtwd1403to1407['forth_month'] == 1]
index = pd.date_range('2014-7-15', '2014-8-23')
redeem_forth = pd.DataFrame((forth_month.loc[:, 'total_redeem_amt']).values, index=index, columns=['total_redeem_amt'])
redeem_forth = redeem_forth.astype('float64')
# 提取全部日期
redeem_data1403to1407 = pd.DataFrame(prtwd1403to1407['total_redeem_amt'])
redeem_data1403to1407 = redeem_data1403to1407.astype('float64')

# 第1周时序图
fig1 = plt.figure()
redeem_first.plot()
plt.title('first_month ts')
# 第2周时序图
fig2 = plt.figure()
redeem_second.plot()
plt.title('second_month ts')
# 第3周时序图
fig3 = plt.figure()
redeem_third.plot()
plt.title('third_month ts')
# 第4周时序图
fig4 = plt.figure()
redeem_forth.plot()
plt.title('forth_month ts')

print('begin ARMA')
'''
warnings.filterwarnings("ignore")
text_file = open("../test/f1234/festival.txt", "w")
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

text_file = open("../test/f1234/first.txt", "w")
for p in range(0, 10):
    for q in range(0, 10):
        if p != q:
            try:
                arma_model = sm.tsa.ARMA(redeem_first, (p, q)).fit()
                text_file.write(
                    'first redeem ARMA(%d,%d)AIC:%f BIC:%f HQOC:%f' % (p, q, arma_model.aic, arma_model.bic, arma_model.hqic))
                text_file.write('\n')
            except:
                continue

text_file.close()

text_file = open("../test/f1234/second.txt", "w")
for p in range(0, 10):
    for q in range(0, 10):
        if p != q:
            try:
                arma_model = sm.tsa.ARMA(redeem_second, (p, q)).fit()
                text_file.write(
                    'second redeem ARMA(%d,%d)AIC:%f BIC:%f HQOC:%f' % (p, q, arma_model.aic, arma_model.bic, arma_model.hqic))
                text_file.write('\n')
            except:
                continue

text_file.close()

text_file = open("../test/f1234/third.txt", "w")
for p in range(0, 10):
    for q in range(0, 10):
        if p != q:
            try:
                arma_model = sm.tsa.ARMA(redeem_third, (p, q)).fit()
                text_file.write(
                    'third redeem ARMA(%d,%d)AIC:%f BIC:%f HQOC:%f' % (p, q, arma_model.aic, arma_model.bic, arma_model.hqic))
                text_file.write('\n')
            except:
                continue

text_file.close()

text_file = open("../test/f1234/forth.txt", "w")
for p in range(0, 10):
    for q in range(0, 10):
        if p != q:
            try:
                arma_model = sm.tsa.ARMA(redeem_forth, (p, q)).fit()
                text_file.write(
                    'forth redeem ARMA(%d,%d)AIC:%f BIC:%f HQOC:%f' % (p, q, arma_model.aic, arma_model.bic, arma_model.hqic))
                text_file.write('\n')
            except:
                continue

text_file.close()
'''
'''
warnings.filterwarnings("ignore")
# redeem_festival_model = sm.tsa.ARMA(redeem_festival, (2, 0)).fit()
redeem_first_model = sm.tsa.ARMA(redeem_first, (4, 2)).fit()
redeem_second_model = sm.tsa.ARMA(redeem_second, (5, 3)).fit()
redeem_third_model = sm.tsa.ARMA(redeem_third, (6, 3)).fit()
redeem_forth_model = sm.tsa.ARMA(redeem_forth, (5, 3)).fit()


# 预测8月
# redeem_festival_predict = redeem_festival_model.predict('2014-08-30', '2014-08-31', dynamic=False)
redeem_first_predict = redeem_first_model.predict('2014-08-01', '2014-08-07', dynamic=False)
redeem_second_predict = redeem_second_model.predict('2014-08-08', '2014-08-16', dynamic=False)
redeem_third_predict = redeem_third_model.predict('2014-08-17', '2014-08-23', dynamic=False)
redeem_forth_predict = redeem_forth_model.predict('2014-08-24', '2014-08-31', dynamic=False)
redeem_predict = ((redeem_first_predict.append(redeem_second_predict)).append(redeem_third_predict)).append(redeem_forth_predict)
print(redeem_predict)

# 画预测图
fig, ax = plt.subplots(figsize=(12, 8))
redeem_predict.plot(ax=ax, label='predict')
prtwd1408['total_redeem_amt'].plot(ax=ax, label='real')
plt.legend()
plt.title('8\'s total_redeem_amt predict')
plt.draw()

# 计算误差
redeem_error = np.divide(np.abs(np.array(prtwd1408['total_redeem_amt']) - np.array(redeem_predict).T),
                         np.array(prtwd1408['total_redeem_amt']))
print('redeem_error\n')
print(redeem_error)
'''
plt.show()
