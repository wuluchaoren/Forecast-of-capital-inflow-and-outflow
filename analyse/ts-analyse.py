import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
pd.options.mode.chained_assignment = None
import analyse.stationarity as test
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
purchase_total = user_balance['total_purchase_amt'].sum()
purchase_total_diff = purchase_total.diff(1)
redeem_total = user_balance['total_redeem_amt'].sum()
redeem_total_diff = redeem_total.diff(1)

# 购买赎回总量ts
fig1 = plt.figure(figsize=(12, 8))
purchase_redeem_total.plot()
plt.title('purchase_redeem_total ts')


# 购买总量ts+diff
# fig2 = plt.figure(figsize=(12, 8))
# ax21 = fig2.add_subplot(211)
# purchase_total.plot()
# ax21.set_title('purchase_total ts')
# ax22 = fig2.add_subplot(212)
# purchase_total_diff.plot()
# ax22.set_title('purchase_total ts_diff')
'''
# 购买总量seasonal_decompose
fig3 = plt.figure()
decomposition = seasonal_decompose(purchase_total, freq=48)
decomposition.plot()
plt.title('purchase_total decomposition')

# 可以分别获得趋势、季节性和随机性
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# 原序列稳定性：
test.test_stationarity(purchase_total, 'origin')
# 一阶差分稳定性：
first_difference = purchase_total.diff(1)
test.test_stationarity(purchase_total.dropna(inplace=False), 'first_difference')
# 季节查分
seasonal_difference = purchase_total.diff(12)
test.test_stationarity(seasonal_difference.dropna(inplace=False), 'seasonal_difference')
# 一阶查分和季节查分合并
seasonal_first_difference = first_difference.diff(12)
test.test_stationarity(seasonal_first_difference.dropna(inplace=False), 'seasonal_first_difference')
'''

'''
Results of Dickey-Fuller Test: origin
Test Statistic                  -1.589880
p-value                          0.488675
#Lags Used                      18.000000
Number of Observations Used    408.000000
Critical Value (1%)             -3.446480
Critical Value (5%)             -2.868650
Critical Value (10%)            -2.570557
dtype: float64

Results of Dickey-Fuller Test: first_difference
Test Statistic                  -1.589880
p-value                          0.488675
#Lags Used                      18.000000
Number of Observations Used    408.000000
Critical Value (1%)             -3.446480
Critical Value (5%)             -2.868650
Critical Value (10%)            -2.570557
dtype: float64

Results of Dickey-Fuller Test: seasonal_difference
Test Statistic                  -4.826324
p-value                          0.000048
#Lags Used                      16.000000
Number of Observations Used    398.000000
Critical Value (1%)             -3.446888
Critical Value (5%)             -2.868829
Critical Value (10%)            -2.570653
dtype: float64

Results of Dickey-Fuller Test: seasonal_first_difference
Test Statistic                -8.737404e+00
p-value                        3.073639e-14
#Lags Used                     1.500000e+01
Number of Observations Used    3.980000e+02
Critical Value (1%)           -3.446888e+00
Critical Value (5%)           -2.868829e+00
Critical Value (10%)          -2.570653e+00
dtype: float64
'''

'''

# 赎回总量ts+diff
# fig4 = plt.figure(figsize=(12, 8))
# ax21 = fig3.add_subplot(211)
# redeem_total.plot()
# ax21.set_title('redeem_total ts')
# ax22 = fig3.add_subplot(212)
# redeem_total_diff.plot()
# ax22.set_title('redeem_total ts_diff')

# 赎回总量seasonal_decompose
fig5 = plt.figure()
decomposition = seasonal_decompose(redeem_total, freq=48)
decomposition.plot()
plt.title('redeem_total decomposition')

# 可以分别获得趋势、季节性和随机性
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# 原序列稳定性：
test.test_stationarity(redeem_total, 'origin')
# 一阶差分稳定性：
first_difference = redeem_total.diff(1)
test.test_stationarity(redeem_total.dropna(inplace=False), 'first_difference')
# 季节查分
seasonal_difference = redeem_total.diff(12)
test.test_stationarity(seasonal_difference.dropna(inplace=False), 'seasonal_difference')
# 一阶查分和季节查分合并
seasonal_first_difference = first_difference.diff(12)
test.test_stationarity(seasonal_first_difference.dropna(inplace=False), 'seasonal_first_difference')
'''


plt.show()
