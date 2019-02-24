import pandas as pd
import numpy  as np
import math
from scipy import signal
import matplotlib.pyplot as plt
import statsmodels.api as sm

pd.options.mode.chained_assignment = None


# 类型转换为分类的数字
def type_to_level(df, col_key, type_dict):  # pandas dataframe,键值,字符串对应的字典
    arr = []
    data_num = len(df)
    for i in range(data_num):
        arr.append(type_dict[df[col_key][i]])

    del df[col_key]
    df[col_key] = arr


# 移动平均图
def draw_trend(timeSeries, size0, size1, title):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size0).mean().rolling(window=size1).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean = timeSeries.ewm(span=size0, min_periods=0, adjust=True, ignore_na=False).mean().ewm(span=size1,
                                                                                                           min_periods=0,
                                                                                                           adjust=True,
                                                                                                           ignore_na=False).mean()

    # timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title(title)


# 功率谱密度
def draw_PSD(x, fs, title):
    plt.figure(facecolor='white')
    f, Pxx_den = signal.periodogram(x, fs)
    f = f[1:len(f)]  # 去除零频分量
    Pxx_den = Pxx_den[1:len(Pxx_den)]  #
    plt.semilogy(1 / f, (Pxx_den) / np.max(Pxx_den) + 1)
    plt.title(title)


# 偏移相关性检测
def offset_corr(x1, x2, max_offset, window_size, title):
    re = list([])
    for i in range(0, 2 * max_offset):
        t1 = x1[max_offset:window_size + max_offset]
        t2 = x2[i:window_size + i]
        re.append((pd.DataFrame(t1)).corrwith(t2).ix[0, 0])
    plt.figure(facecolor='white')
    plt.plot(list(range(-max_offset, max_offset)), re)
    plt.xlabel("offset")
    plt.ylabel("r")
    plt.title(title)
    return re


# 读取数据文件
# 用户数据:id 性别 城市 星座 
user_profile_table = pd.read_csv(r"../.../data/user_profile_table.csv", sep=',', engine='python', encoding='utf-8')
# 用户申购赎回数数据
user_balance_table = pd.read_csv(r"../.../data/user_balance_table.csv", sep=',', engine='python', encoding='utf-8',
                                 parse_dates=['report_date'])
# 收益率：日期 万分收益 七日年化收益
mfd_day_share_interest = pd.read_csv(r"../.../data/mfd_day_share_interest.csv", sep=',', engine='python', encoding='utf-8',
                                     parse_dates=['mfd_date'])
# 银行拆放利率:日期 隔夜利率（%）1周利率（%）2周利率（%）1个月利率（%）3个月利率（%）6个月利率（%）9个月利率（%）
mfd_bank_shibor = pd.read_csv(r"../.../data/mfd_bank_shibor.csv", sep=',', engine='python', encoding='utf-8',
                              parse_dates=['mfd_date'])

# user_balance_table 数据缺失值处理 填充0(众数)
user_balance_table = user_balance_table.fillna(0)

# 数据分析

# 购买赎回总量分析
user_balance = user_balance_table.groupby(['report_date'])
purchase_redeem_total = user_balance['total_purchase_amt', 'total_redeem_amt'].sum()

# purchase_redeem_total.plot()
bound = 50000
# 单笔大于或小于1W的购买赎回分析
user_balance_op_more_than_bound = user_balance_table[
    (user_balance_table['total_purchase_amt'] >= bound) | (user_balance_table['total_redeem_amt'] >= bound)]
time_group_more_than_bound = user_balance_op_more_than_bound.groupby(['report_date'])
purchase_redeem_total_more_than_bound = time_group_more_than_bound['total_purchase_amt', 'total_redeem_amt'].sum()

purchase_redeem_total_more_than_bound = purchase_redeem_total_more_than_bound.rename(
    columns={'total_purchase_amt': 'total_purchase_amt_more_than_bound'})
purchase_redeem_total_more_than_bound = purchase_redeem_total_more_than_bound.rename(
    columns={'total_redeem_amt': 'total_redeem_amt_more_than_bound'})
# purchase_redeem_total_more_than_bound.plot()


user_balance_op_less_than_bound = user_balance_table[
    (user_balance_table['total_purchase_amt'] < bound) & (user_balance_table['total_redeem_amt'] < bound)]
time_group_less_than_bound = user_balance_op_less_than_bound.groupby(['report_date'])
purchase_redeem_total_less_than_bound = time_group_less_than_bound['total_purchase_amt', 'total_redeem_amt'].sum()

purchase_redeem_total_less_than_bound = purchase_redeem_total_less_than_bound.rename(
    columns={'total_purchase_amt': 'total_purchase_amt_less_than_bound'})
purchase_redeem_total_less_than_bound = purchase_redeem_total_less_than_bound.rename(
    columns={'total_redeem_amt': 'total_redeem_amt_less_than_bound'})
# purchase_redeem_total_less_than_bound.plot()

# 合并三个Data Frame
purchase_redeem_total_concat_result = pd.concat([purchase_redeem_total, purchase_redeem_total_more_than_bound], axis=1)
purchase_redeem_total_concat_result = pd.concat(
    [purchase_redeem_total_concat_result, purchase_redeem_total_less_than_bound], axis=1)
purchase_redeem_total_concat_result.plot()

# 做滑动平均和加权滑动平均
draw_trend(purchase_redeem_total['total_purchase_amt'], 7, 30, 'purchase_Rolling Mean')
draw_trend(purchase_redeem_total['total_redeem_amt'], 7, 30, 'redeem_Rolling Mean ')
draw_trend(purchase_redeem_total['total_purchase_amt'] - purchase_redeem_total['total_redeem_amt'], 7, 30,
           'purchase_cut_redeem_Rolling Mean ')
plt.draw()

# 谱分析
draw_PSD(purchase_redeem_total['total_purchase_amt'].tolist(), 1, "total_purchase Power Spectrum")
draw_PSD(purchase_redeem_total['total_redeem_amt'].tolist(), 1, "total_redeem_amt Power Spectrum")
draw_PSD((purchase_redeem_total['total_purchase_amt'] - purchase_redeem_total['total_redeem_amt']).tolist(), 1,
         "total_purchase_cut_redeem_amt Power Spectrum")
plt.draw()
# 收益率绘制
time_mfd_day_share_interest = mfd_day_share_interest.groupby(['mfd_date'])
share_interest = time_mfd_day_share_interest['mfd_daily_yield', 'mfd_7daily_yield'].sum()
share_interest.plot()
plt.draw()
# 相关性分析

# 计算星期列
date = pd.DataFrame(purchase_redeem_total.index)
date['day_of_week'] = date['report_date'].dt.dayofweek
tt_date = date.groupby(['report_date'])
tt_date = tt_date['day_of_week'].sum()

t_mfd_bank_shibor = (mfd_bank_shibor.groupby(['mfd_date']))
time_mfd_bank_shibor = t_mfd_bank_shibor['Interest_O_N'].sum()
time_mfd_bank_shibor.fillna(method='pad')

corr_pd = pd.concat([purchase_redeem_total, share_interest, time_mfd_bank_shibor, tt_date], axis=1)
corr_pd = corr_pd.fillna(method='pad')
corr_table = corr_pd.corr()
print(corr_table)
corr_table.to_csv('../test/analyse/corr.csv', index=True, sep=',')

# 偏移相关性分析
# 赎回和申购的偏移相关性分析
offset_corr(corr_pd['total_purchase_amt'], corr_pd['total_redeem_amt'], 90, 200, "total_purchase total_redeem corr")
offset_corr(corr_pd['total_purchase_amt'], corr_pd['mfd_7daily_yield'], 90, 200,
            "total_purchase mfd_7daily_yield corr")
offset_corr(corr_pd['total_purchase_amt'], corr_pd['Interest_O_N'], 90, 200, "total_purchase Interest_O_N corr")
offset_corr(corr_pd['total_redeem_amt'], corr_pd['Interest_O_N'], 90, 200, "total_redeem_amt Interest_O_N corr")
offset_corr(corr_pd['total_redeem_amt'], corr_pd['mfd_7daily_yield'], 90, 200,
            "total_redeem_amt mfd_7daily_yield corr")
offset_corr(corr_pd['total_redeem_amt'], corr_pd['mfd_7daily_yield'], 120, 140,
            "total_redeem_amt mfd_7daily_yield corr 120d offset")

offset_corr(corr_pd['total_purchase_amt'], corr_pd['day_of_week'], 120, 140, "total_purchase_amt day_of_week corr ")
offset_corr(corr_pd['total_redeem_amt'], corr_pd['day_of_week'], 120, 140, "total_redeem_amt day_of_week corr ")

# 去除首部90多个数据的相关性分析
corr_head_rm_table = corr_pd.tail(300).corr()
print(corr_head_rm_table)
corr_table.to_csv('../test/analyse/corr_head_rm.csv', index=True, sep=',')

# 去除尾部90多个数据的相关性分析
corr_tail_rm_table = corr_pd.head(300).corr()
print(corr_tail_rm_table)
corr_table.to_csv('../test/analyse/corr_tail_rm.csv', index=True, sep=',')

# 去除首尾部90多个数据的相关性分析
corr_head_tail_rm_table = corr_pd.head(300).tail(200).corr()
print(corr_head_tail_rm_table)
corr_table.to_csv('../test/analyse/corr_head_tail_rm.csv', index=True, sep=',')

# stl分解
purchase_tsr = sm.tsa.seasonal_decompose(purchase_redeem_total.total_purchase_amt)

purchase_trend = purchase_tsr.trend
purchase_seasonal = purchase_tsr.seasonal
purchase_residual = purchase_tsr.resid

purchase_tsr.plot()
plt.title('total_purchase_amt stl')
plt.draw()

redeem_tsr = sm.tsa.seasonal_decompose(purchase_redeem_total.total_redeem_amt)
redeem_trend = redeem_tsr.trend
redeem_seasonal = redeem_tsr.seasonal
redeem_residual = redeem_tsr.resid

redeem_tsr.plot()
plt.title('total_redeem_amt stl')
plt.draw()

plt.show()
