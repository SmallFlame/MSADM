import matplotlib.pyplot as plt
import numpy as np

# 假设我们有两个组，每个组有四个类别
N = 2  # 类别数量
groups = ('Group1', 'Group2', 'Group3', 'Group4')  # 组名
indices = np.arange(N)  # 类别索引
width = 0.2  # 柱子的宽度

# 为每个组生成一些随机数据

# lenw = 130
# ytitle = 'Anomaly Detection Accuracy'
# AnomalyBERT = [86.78,82.57]
# SR_CNN = [87.89,81.26]
# CL_MPPCA = [86.56,81.64]
# LSTM_transformer = [88.87,83.37]
# MSADM = [95.61,91.46]

lenw = 140
ytitle = 'Anomaly Detection Accuracy'
MobilePhone= [97.26,90.21]
Vehicle = [93.36, 89.45]
UAV = [83.26, 89.67]
BASESTATION = [78.02, 89.54]

# 创建分组柱状图
colors = {
    'MSADM': '#49759C',  # 深蓝色
    'SR_CNN': '#77AADD',  # 亮蓝色
    'CL_MPPCA': '#336699',  # 钴蓝色
    'AnomalyBERT': '#BBDDEE',  # 淡蓝色
    'LSTM_transformer': '#224466'  # 海军蓝
}

fig, ax = plt.subplots()
rects1 = ax.bar(indices, MobilePhone, width, label='Mobile Phone', color=colors['MSADM'])
rects2 = ax.bar(indices + width, Vehicle, width, label='Vehicle', color=colors['SR_CNN'])
rects3 = ax.bar(indices + 2 * width, UAV, width, label='UAV', color=colors['CL_MPPCA'])
rects4 = ax.bar(indices + 3 * width, BASESTATION, width, label='Base Station', color=colors['AnomalyBERT'])

ax.set_ylim(0,lenw)
ax.set_yticks(np.arange(0, 110, 10))

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)  # 调整字体大小为8

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

# 添加标题和标签
ax.set_xlabel('Train Data Type')
ax.set_ylabel(ytitle)
ax.set_xticks(indices + width * 1.5)  # 计算标签的位置
ax.set_xticklabels(('No Rule-based', 'Rule-based'))
ax.legend()

# 显示图形
plt.savefig('device.pdf', dpi=600, bbox_inches='tight')
plt.show()
