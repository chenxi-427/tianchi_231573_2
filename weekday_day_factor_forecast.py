import pandas as pd
import numpy as np

# 读取数据
usecols = ['report_date', 'total_purchase_amt', 'total_redeem_amt']
df = pd.read_csv('user_balance_table.csv', usecols=usecols, encoding='utf-8')
df['report_date'] = pd.to_datetime(df['report_date'], format='%Y%m%d')

# 筛选训练区间
train_start = '2014-03-01'
train_end = '2014-08-31'
df_train = df[(df['report_date'] >= train_start) & (df['report_date'] <= train_end)]
summary = df_train.groupby('report_date').sum().reset_index()

# 添加weekday和day
summary['weekday'] = summary['report_date'].dt.weekday  # 0=周一
summary['day'] = summary['report_date'].dt.day

# 历史均值
purchase_mean = summary['total_purchase_amt'].mean()
redeem_mean = summary['total_redeem_amt'].mean()

# weekday均值
purchase_weekday = summary.groupby('weekday')['total_purchase_amt'].mean()
redeem_weekday = summary.groupby('weekday')['total_redeem_amt'].mean()

# day均值
purchase_day = summary.groupby('day')['total_purchase_amt'].mean()
redeem_day = summary.groupby('day')['total_redeem_amt'].mean()

# 预测2014-09-01到2014-09-30
dates = pd.date_range('2014-09-01', '2014-09-30')
forecast = []
for d in dates:
    wd = d.weekday()
    daynum = d.day
    # 申购预测 = weekday因子 + day因子 - 历史均值
    purchase_pred = purchase_weekday[wd] + purchase_day.get(daynum, purchase_mean) - purchase_mean
    redeem_pred = redeem_weekday[wd] + redeem_day.get(daynum, redeem_mean) - redeem_mean
    forecast.append([d.strftime('%Y%m%d'), purchase_pred, redeem_pred])

# 写入csv
forecast_df = pd.DataFrame(forecast, columns=['report_date', 'purchase', 'redeem'])
forecast_df.to_csv('weekday_day_factor_forecast_09.csv', index=False, header=False, encoding='utf-8')
print('周期因子预测结果已保存到 weekday_day_factor_forecast_09.csv') 