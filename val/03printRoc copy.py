import pandas as pd  
from sklearn.metrics import roc_curve, auc  
import matplotlib.pyplot as plt  
  
modellist = [  
    "out/test/TransformerRE/res20.csv",  
    "out/test/LSTMTransformer/res60.csv",  
    "out/test/LSTMRE/res60.csv",  
    "out/test/CNNRE/res60.csv",  
    "out/test/AttentionLstmRE/res60.csv"  
]  
  
labels = [  
    'MSADM',  
    'LSTM-transformer',  
    'CL-MPPCA',  
    'SR-CNN',  
    'AnomalyBERT'  
]  
  
colors = ['tab:purple', 'tab:red', 'tab:orange', 'tab:blue', 'tab:green']  
  
plt.figure()  
lw = 2  
  
for model_file, label, color in zip(modellist, labels, colors):  
    df = pd.read_csv(model_file)  
    y_true = df['out_dectlabel']  # 假设这是实际标签列  
    y_scores = df['out_dectlist']  # 假设这是预测得分列（可能需要阈值化或已经是概率）  
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)  
    roc_auc = auc(fpr, tpr)  
      
    plt.plot(fpr, tpr, color=color, lw=lw,  
             label=f'{label} (AUC = {roc_auc:.2f})')  
  
# 绘制对角线  
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  
  
# 设置图的极限  
plt.xlim([0.0, 1.0])  
plt.ylim([0.0, 1.05])  
  
# 设置轴标签和标题  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Multiple Receiver Operating Characteristic Curves')  
  
# 显示图例  
plt.legend(loc="lower right")  
  
# 显示图像  
plt.show()  
  
# 如果需要保存图像  
plt.savefig("multi-roc.pdf", bbox_inches='tight', dpi=600)