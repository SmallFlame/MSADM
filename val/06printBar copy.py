import matplotlib.pyplot as plt
import numpy as np
N = 2
groups = ('Group1', 'Group2', 'Group3', 'Group4')
indices = np.arange(N)
width = 0.2

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

colors = {
    'MSADM': '#49759C', 
    'SR_CNN': '#77AADD',
    'CL_MPPCA': '#336699',  
    'AnomalyBERT': '#BBDDEE', 
    'LSTM_transformer': '#224466'
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
                    fontsize=8)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

ax.set_xlabel('Train Data Type')
ax.set_ylabel(ytitle)
ax.set_xticks(indices + width * 1.5) 
ax.set_xticklabels(('No Rule-based', 'Rule-based'))
ax.legend()

plt.savefig('device.pdf', dpi=600, bbox_inches='tight')
plt.show()
