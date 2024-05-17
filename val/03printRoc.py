import time
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.metrics import roc_curve, auc
import torch  
from model.modelutil.MSADM import test as m
from model.modelutil.LstmModel import test as l
from model.modelutil.AttentionLstm import test as a
from model.modelutil.LSTMTransformerModel import test as t
from model.modelutil.CNNModel import test as c
from model.config import trainConfig
tra_math_path = trainConfig["tra_math_path"]
tra_ruler_path = trainConfig["tra_ruler_path"]
val_math_path = trainConfig["val_math_path"]
val_ruler_path = trainConfig["val_ruler_path"]
learning_rate = trainConfig["learning_rate"]
batch_size = trainConfig["batch_size"]
epoch_num = trainConfig["epoch_num"]
input_dim = 22
length = 96
output_dim = 7
# transformer层数
num_layers = 2
dropout = 0.3

device = torch.device( "cpu") 
# "cuda:1" if torch.cuda.is_available() else
result_path = f'out/test/TransformerRE/res.csv'
modellist = ["out/model/AttentionLstmRE/model60.pkl",
             "out/model/CNNRE/model60.pkl",
             "out/model/LSTMRE/model60.pkl",
             "out/model/TransformerRE/model60.pkl",
             "out/model/LSTMTransformer/model60.pkl"]

ap,at = [],[]
cp,ct = [],[]
lp,lt = [],[]
mp,mt = [],[]
tp,tt = [],[]

ap,at = a(val_math_path,val_ruler_path,result_path,modellist[0])
cp,ct = c(val_math_path,val_ruler_path,result_path,modellist[1])
lp,lt = l(val_math_path,val_ruler_path,result_path,modellist[2])
mp,mt = m(val_math_path,val_ruler_path,result_path,modellist[3])
tp,tt = t(val_math_path,val_ruler_path,result_path,modellist[4])

fpr1, tpr1, thresholds1 = roc_curve(ap,at)  
roc_auc1 = auc(fpr1, tpr1)  
  
fpr2, tpr2, thresholds2 = roc_curve(cp,ct)  
roc_auc2 = auc(fpr2, tpr2)  
  
fpr3, tpr3, thresholds3 = roc_curve(lp,lt )  
roc_auc3 = auc(fpr3, tpr3)  
fpr4, tpr4, thresholds4 = roc_curve( mp,mt)  
roc_auc4 = auc(fpr4, tpr4)  
fpr5, tpr5, thresholds4 = roc_curve(tp,tt)  
roc_auc5 = auc(fpr5, tpr5)  
  
# 绘制ROC曲线  
plt.figure()  
lw = 2  

# 第一组数据  
plt.plot(fpr1, tpr1,  lw=lw,   
         label=f'AnomalyBERT')  
# 第二组数据  
plt.plot(fpr2, tpr2,  lw=lw,   
         label=f'SR-CNN')  
  
  
# 第三组数据  
plt.plot(fpr3, tpr3, lw=lw,   
         label=f'CL-MPPCA')  
  
# 第四组数据  
plt.plot(fpr4, tpr4,  lw=lw,   
         label=f'MSADM')  
# 第四组数据  
plt.plot(fpr5, tpr5,  lw=lw,   
         label=f'LSTM-transformer')  
  
# 绘制对角线  
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  
  
# 设置图的极限  
plt.xlim([0.0, 1.0])  
plt.ylim([0.0, 1.05])  
  
# 设置轴标签和标题  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
# plt.title('Multiple Receiver Operating Characteristic Curves')  
  
# 显示图例  
plt.legend(loc="lower right")  
  
# 显示图像  
plt.show()  
  
# 如果需要保存图像  
plt.savefig("multi-roc.pdf", bbox_inches='tight', dpi=600)

