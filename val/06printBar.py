import matplotlib.pyplot as plt
import numpy as np

N = 2 
groups = ('Group1', 'Group2', 'Group3', 'Group4')
indices = np.arange(N)
width = 0.15

lenw = 140
ytitle = 'Anomaly Detection Accuracy'
AnomalyBERT = [86.78,82.57]
SR_CNN = [87.89,81.26]
CL_MPPCA = [86.56,81.64]
LSTM_transformer = [88.87,83.37]
MSADM = [91.61,88.46]

# lenw = 110
# ytitle = 'Anomaly Classification Accuracy'
# AnomalyBERT= [66.53,62.21]
# SR_CNN = [59.36, 53.89]
# CL_MPPCA = [69.69, 61.67]
# LSTM_transformer = [72.02, 68.54]
# MSADM = [76.73, 71.86]

colors = {
    'MSADM': '#49759C',
    'SR_CNN': '#77AADD',
    'CL_MPPCA': '#336699',
    'AnomalyBERT': '#BBDDEE',
    'LSTM_transformer': '#224466'
}

fig, ax = plt.subplots()
rects1 = ax.bar(indices, MSADM, width, label='MSADM', color=colors['MSADM'])
rects2 = ax.bar(indices + width, SR_CNN, width, label='SR-CNN', color=colors['SR_CNN'])
rects3 = ax.bar(indices + 2 * width, CL_MPPCA, width, label='CL-MPPCA', color=colors['CL_MPPCA'])
rects4 = ax.bar(indices + 3 * width, AnomalyBERT, width, label='AnomalyBERT', color=colors['AnomalyBERT'])
rects5 = ax.bar(indices + 4 * width, LSTM_transformer, width, label='LSTM-transformer', color=colors['LSTM_transformer'])

ax.set_ylim(0,lenw)
ax.set_yticks(np.arange(0, 110, 10))

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)

ax.set_xlabel('Node Number')
ax.set_ylabel(ytitle)
ax.set_xticks(indices + width * 2)
ax.set_xticklabels(('Node 9-12', 'Node 15-17'))
ax.legend()

plt.savefig('dbar.pdf', dpi=600, bbox_inches='tight')
