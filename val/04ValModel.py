import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score  

# obstacle 1 1
# out 2 1
# normal 0 0
# nodedown 3 1
# congest 4 1
# malicious 5 1
# appdown 6 1

def getaccuracy(y_true, y_pred):
    # 使用accuracy_score计算准确率  
    accuracy = accuracy_score(y_true, y_pred)  
    print("accuracy:",accuracy)
    return accuracy

def calculate_metrics(y_true, y_pred):  
    """  
    Calculate recall, FNR, and FPR.  
      
    :param y_true: List of true labels.  
    :param y_pred: List of predicted labels.  
    :return: recall, fnr, fpr  
    """  
    # 初始化计数器  
    TP, FP, TN, FN = 0, 0, 0, 0  
      
    # 计算TP, FP, TN, FN  
    for y_t, y_p in zip(y_true, y_pred):  
        if y_t == 1 and y_p == 1:  
            TP += 1  
        elif y_t == 0 and y_p == 1:  
            FP += 1  
        elif y_t == 0 and y_p == 0:  
            TN += 1  
        elif y_t == 1 and y_p == 0:  
            FN += 1  
      
    # 计算召回率、假负率和假正率  
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0  
    fnr = FN / (TP + FN) if (TP + FN) > 0 else 0  
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0  
    print( f"召回率{recall}、假负率{fnr}和假正率{fpr}  " )
    return recall,fnr,fpr
      
def draw_confusion_matrix(label_true, label_pred, label_name, pdf_save_path=None, title="Confusion Matrix", dpi=300):
    """

    @param label_true: 真实标签
    @param label_pred: 预测标签
    @param label_name: 标签名字
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi:至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)

def getrate(result_path,val_result,num):
    results_df = pd.DataFrame(columns=['Epoch', 'Accuracy', 'Recall', 'FNR', 'FPR',
                                    'Dec', 'RecallD', 'FNRD', 'FPRD']) 
    for i in range(num):
        df = pd.read_csv(f"{result_path}/res{i+1}.csv")
        y_pred = df['outputslist'].to_numpy()  
        y_true = df['outputslabels'].to_numpy()
        y_pred_d = df['out_dectlist'].to_numpy()  
        y_true_d = df['out_dectlabel'].to_numpy()  
        recall1,fnr1,fpr1 = calculate_metrics(y_true,y_pred)
        recall2,fnr2,fpr2 = calculate_metrics(y_true_d, y_pred_d)
        acc1 = getaccuracy(y_true, y_pred)
        acc2 = getaccuracy(y_true_d, y_pred_d)
        loss1 = df['lossC'].to_numpy().mean()
        loss2 = df['lossD'].to_numpy().mean()  
        loss = loss1+loss2
        results_df = results_df.append({'Accuracy': acc1, 'Recall': recall1,
                                'FNR': fnr1, 'FPR': fpr1,'Dec':acc2, 
                                'RecallD':recall2, 'FNRD':fnr2, 'FPRD':fpr2,'loss':loss}, ignore_index=True)
    results_df.to_csv(val_result, index=False)

# TransformerRE
model_name = 'TransformerRE'
getrate(f'out/test/{model_name}',f'out/test/{model_name}/01result.csv',70)