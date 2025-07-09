import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# 设置中文字体，防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 只读取需要的列，节省内存
usecols = ['report_date', 'total_purchase_amt', 'total_redeem_amt']
df = pd.read_csv('user_balance_table.csv', usecols=usecols, encoding='utf-8')

# 日期格式转换
date_fmt = '%Y%m%d'
df['report_date'] = pd.to_datetime(df['report_date'], format=date_fmt)

# 筛选2014-03到2014-08的数据
start_date = '2014-03-01'
end_date = '2014-08-31'
df_filtered = df[(df['report_date'] >= start_date) & (df['report_date'] <= end_date)]

# 按report_date分组求和
summary = df_filtered.groupby('report_date').sum().reset_index()

# ADF平稳性检验函数
def adf_test(series, name):
    print(f'对{name}进行ADF平稳性检验:')
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print(f'临界值 {key}: {value}')
    print('-' * 40)

# 对总申购金额和总赎回金额分别进行ADF检验
adf_test(summary["total_purchase_amt"], "总申购金额")
adf_test(summary["total_redeem_amt"], "总赎回金额")

# ARIMA预测未来30天（2014-09）
purchase_series = summary['total_purchase_amt']
redeem_series = summary['total_redeem_amt']

# 申购金额ARIMA建模
purchase_model = ARIMA(purchase_series, order=(7, 0, 7))
purchase_fit = purchase_model.fit()
purchase_forecast = purchase_fit.forecast(steps=30)

# 赎回金额ARIMA建模
redeem_model = ARIMA(redeem_series, order=(7, 1, 5))
redeem_fit = redeem_model.fit()
redeem_forecast = redeem_fit.forecast(steps=30)

# 生成2014-09日期
last_date = summary['report_date'].max()
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')

# 结果保存到DataFrame
result_df = pd.DataFrame({
    'report_date': forecast_dates,
    'purchase': purchase_forecast,
    'redeem': redeem_forecast
})

# 输出到csv
result_df_out = result_df.copy()
result_df_out['report_date'] = result_df_out['report_date'].dt.strftime('%Y%m%d')
result_df_out.to_csv('prophet_prediction_results.csv', index=False, header=False)
print('预测结果已保存到 prophet_prediction_results.csv')

# 画图（历史+预测）
plt.figure(figsize=(12, 6))
plt.plot(summary['report_date'], summary['total_purchase_amt'], label='历史总申购金额')
plt.plot(summary['report_date'], summary['total_redeem_amt'], label='历史总赎回金额')
plt.plot(result_df['report_date'], result_df['purchase'], label='预测总申购金额', linestyle='--')
plt.plot(result_df['report_date'], result_df['redeem'], label='预测总赎回金额', linestyle='--')
plt.xlabel('日期')
plt.ylabel('金额')
plt.title('2014年3月至9月每日总申购与赎回金额（含预测）')
plt.legend()
plt.tight_layout()
plt.show() 