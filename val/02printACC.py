import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker  
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [8, 5]
fig = plt.figure(figsize=(10, 6)) 
# cnn = 'out/test/CNNRE/01result.csv'
# lstm = 'out/test/LSTMRE/01result.csv'
# transformer = 'out/test/TransformerRE/01result.csv'
# attentionlstm ='out/test/AttentionLstmRE/01result.csv'
# transformerlstm = 'out/test/LSTMTransformer/01result.csv'

cnn = 'out/test/CNNRE/01result.csv'
lstm = 'out/test/LSTMRE/01result.csv'
transformer = 'out/test/TransformerRE/01result.csv'
attentionlstm ='out/test/AttentionLstmRE/01result.csv'
transformerlstm = 'out/test/LSTMTransformer/01result.csv'

SR_CNN = pd.read_csv(cnn)
CL_MPPCA = pd.read_csv(lstm)
AnomalyBERT = pd.read_csv(attentionlstm)
LSTM_transformer = pd.read_csv(transformerlstm)
MSADM = pd.read_csv(transformer)

# SR_CNN =SR_CNN.iloc[:60,1]
# CL_MPPCA =CL_MPPCA.iloc[:60,1]
# AnomalyBERT =AnomalyBERT.iloc[:60,1]
# LSTM_transformer =LSTM_transformer.iloc[:60,1]
# MSADM = MSADM.iloc[0:60,1]
SR_CNN =SR_CNN.iloc[:60,5].tolist()
CL_MPPCA =CL_MPPCA.iloc[:60,5].tolist()
AnomalyBERT =AnomalyBERT.iloc[:60,5].tolist()
LSTM_transformer =LSTM_transformer.iloc[:60,5].tolist()
MSADM = MSADM.iloc[0:60,5]
fig, ax = plt.subplots()  
x=range(1,61)
ax.plot(x, SR_CNN,label='SR-CNN')  
ax.plot(x, CL_MPPCA,label='CL-MPPCA')  
ax.plot(x, AnomalyBERT,label='AnomalyBERT') 
ax.plot(x, LSTM_transformer,label='LSTM-transformer') 
ax.plot(x, MSADM,label='MSADM')  

# ax.plot(x, lstmLoss9C,label='Node9to12 Classify')  
# ax.plot(x, lstmLoss9D,label='Node9to12 Detect')  
# # ax.plot(x, lstmLoss9C5,label='Node9to125 Classify')  
# # ax.plot(x, lstmLoss9D5,label='Node9to125 Detect')  
# ax.plot(x, lstmLoss17C,label='Node17to20 Classify')  
# ax.plot(x, lstmLoss17D,label='Node17to20 Detect')  
# print(y)
# 设置y轴的刻度  
xticks = np.arange(0,66, 6) # 设置刻度的范围和间隔  
xticks[0] = 1
# yticks = np.arange(0.3,0.85, 0.05)
yticks = np.arange(0.75,1.025, 0.025) # 设置刻度的范围和间隔  
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}  # 修改字体样式
plt.rc('font', **font)
ax.set_yticks(yticks)  
ax.set_yticklabels(yticks)
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
ax.set_xticks(xticks)
ax.set_xticklabels(xticks)
legend = plt.legend()
legend.get_texts()[0].set_fontsize(12)  
legend.get_texts()[0].set_fontname('Times New Roman')  
legend.get_texts()[1].set_fontsize(12)  
legend.get_texts()[1].set_fontname('Times New Roman')  
legend.get_texts()[2].set_fontsize(12)  
legend.get_texts()[2].set_fontname('Times New Roman')  
legend.get_texts()[3].set_fontsize(12)  
legend.get_texts()[3].set_fontname('Times New Roman') 
# legend.get_texts()[4].set_fontsize(12)  
# legend.get_texts()[4].set_fontname('Times New Roman') 
# legend.get_texts()[4].set_fontsize(12)  
# legend.get_texts()[4].set_fontname('Times New Roman') 
plt.xlabel('Epoch', fontsize=12,fontname='Times New Roman')
# plt.ylabel('Anomaly Classification Accuracy', fontsize=12,fontname='Times New Roman') 
plt.ylabel('Anomaly Detection Accuracy', fontsize=12,fontname='Times New Roman') 
# 显示图表  
plt.savefig('acc1.pdf',dpi=600, bbox_inches='tight')
