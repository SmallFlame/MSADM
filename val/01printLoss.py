import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker  
plt.rcParams['font.family'] = 'Times New Roman'
  
plt.rcParams['figure.figsize'] = [8, 5]
fig = plt.figure(figsize=(3,3)) 
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
SR_CNN =  SR_CNN.iloc[:60,-1].tolist()
CL_MPPCA =CL_MPPCA.iloc[:60,-1].tolist()
AnomalyBERT =AnomalyBERT.iloc[:60,-1].tolist()
LSTM_transformer =LSTM_transformer.iloc[:60,-1].tolist()
MSADM = MSADM.iloc[0:60,-1]
fig, ax = plt.subplots()  
x=range(1,61)
ax.plot(x, SR_CNN,label='SR-CNN')  
ax.plot(x, CL_MPPCA,label='CL-MPPCA')  
ax.plot(x, AnomalyBERT,label='AnomalyBERT') 
ax.plot(x, LSTM_transformer,label='LSTM-transformer') 
ax.plot(x, MSADM,label='MSADM')  
xticks = np.arange(0,66, 6)
xticks[0] = 1
yticks = np.arange(0.5,3.25, 0.25)
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}
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
legend.get_texts()[4].set_fontsize(12)  
legend.get_texts()[4].set_fontname('Times New Roman') 
plt.xlabel('Epoch', fontsize=12,fontname='Times New Roman')
plt.ylabel('Cross Entropy Loss', fontsize=12,fontname='Times New Roman') 
# 显示图表  
plt.savefig('loss.pdf',dpi=1200, bbox_inches='tight')
