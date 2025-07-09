import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体，防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
usecols = ['report_date', 'total_purchase_amt', 'total_redeem_amt']
df = pd.read_csv('user_balance_table.csv', usecols=usecols, encoding='utf-8')
df['report_date'] = pd.to_datetime(df['report_date'], format='%Y%m%d')

# 筛选2014-03-01到2014-08-31
start_date = '2014-03-01'
end_date = '2014-08-31'
df_period = df[(df['report_date'] >= start_date) & (df['report_date'] <= end_date)]

# 按日期聚合
summary = df_period.groupby('report_date').sum().reset_index()

# 添加星期几列（0=周一, 6=周日）
summary['weekday'] = summary['report_date'].dt.weekday
weekday_map = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'}
summary['weekday_cn'] = summary['weekday'].map(weekday_map)

# 按星期几分组，准备箱线图数据
purchase_data = [summary[summary['weekday'] == i]['total_purchase_amt'] for i in range(7)]
redeem_data = [summary[summary['weekday'] == i]['total_redeem_amt'] for i in range(7)]
labels = [weekday_map[i] for i in range(7)]

# 画申购金额箱线图
plt.figure(figsize=(10, 5))
plt.boxplot(purchase_data, labels=labels, patch_artist=True)
plt.xlabel('星期')
plt.ylabel('申购金额')
plt.title('2014年3月至8月各星期申购金额箱线图')
plt.tight_layout()
plt.show()

# 画赎回金额箱线图
plt.figure(figsize=(10, 5))
plt.boxplot(redeem_data, labels=labels, patch_artist=True)
plt.xlabel('星期')
plt.ylabel('赎回金额')
plt.title('2014年3月至8月各星期赎回金额箱线图')
plt.tight_layout()
plt.show() 