import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 设置中文字体，防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
usecols = ['report_date', 'total_purchase_amt', 'total_redeem_amt']
df = pd.read_csv('user_balance_table.csv', usecols=usecols, encoding='utf-8')
df['report_date'] = pd.to_datetime(df['report_date'], format='%Y%m%d')

# 筛选训练集
train_start = '2014-03-01'
train_end = '2014-08-31'
df_train = df[(df['report_date'] >= train_start) & (df['report_date'] <= train_end)]
summary = df_train.groupby('report_date').sum().reset_index()

# 节假日设置
holidays = pd.DataFrame({
    'holiday': ['qingming'] * 3 + ['labour'] * 3 + ['duanwu'] * 3 + ['zhongqiu'] * 3,
    'ds': pd.to_datetime([
        '2014-04-05', '2014-04-06', '2014-04-07',
        '2014-05-01', '2014-05-02', '2014-05-03',
        '2014-05-31', '2014-06-01', '2014-06-02',
        '2014-09-06', '2014-09-07', '2014-09-08'
    ])
})

# 申购金额Prophet建模
purchase_df = summary[['report_date', 'total_purchase_amt']].rename(columns={'report_date': 'ds', 'total_purchase_amt': 'y'})
purchase_model = Prophet(holidays=holidays, yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
purchase_model.fit(purchase_df)

# 赎回金额Prophet建模
redeem_df = summary[['report_date', 'total_redeem_amt']].rename(columns={'report_date': 'ds', 'total_redeem_amt': 'y'})
redeem_model = Prophet(holidays=holidays, yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
redeem_model.fit(redeem_df)

# 预测区间（训练集+9月）
all_dates = pd.date_range(start='2014-03-01', end='2014-09-30', freq='D')
predict_df = pd.DataFrame({'ds': all_dates})

# 申购预测
purchase_forecast = purchase_model.predict(predict_df)
# 赎回预测
redeem_forecast = redeem_model.predict(predict_df)

# 9月结果整理
result = pd.DataFrame({
    'report_date': predict_df['ds'].dt.strftime('%Y%m%d'),
    'purchase': purchase_forecast['yhat'],
    'redeem': redeem_forecast['yhat']
})
result_09 = result[(result['report_date'] >= '20140901') & (result['report_date'] <= '20140930')]
result_09.to_csv('prophet_prediction_results_09.csv', index=False, header=False, encoding='utf-8')
print('Prophet预测结果已保存到 prophet_prediction_results_09.csv')

# 可视化：历史、Prophet预测（训练区间+9月）
plt.figure(figsize=(12, 6))
plt.plot(summary['report_date'], summary['total_purchase_amt'], label='历史总申购金额', color='blue')
plt.plot(summary['report_date'], summary['total_redeem_amt'], label='历史总赎回金额', color='green')
plt.plot(predict_df['ds'], purchase_forecast['yhat'], label='Prophet预测申购金额', linestyle='--', color='red')
plt.plot(predict_df['ds'], redeem_forecast['yhat'], label='Prophet预测赎回金额', linestyle='--', color='orange')
plt.xlabel('日期')
plt.ylabel('金额')
plt.title('2014年3月至9月每日总申购与赎回金额（Prophet预测）')
plt.legend()
plt.tight_layout()
plt.show() 