import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score  
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
    # plt.title(title)
    plt.xlabel("Predict Label")
    plt.ylabel("Truth Label")
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

# obstacle 1 1
# out 2 1
# normal 0 0
# nodedown 3 1
# congest 4 1
# malicious 5 1
# appdown 6 1

# AttentionLstmRE
data_path = "out/test/TransformerRE/res60.csv"
df = pd.read_csv(data_path)
# y_pred = df['outputslist'].to_numpy()  
# y_true = df['outputslabels'].to_numpy()
# label_name = ["normal","obstacle","out","nodedown","congest","malicious","appdown"]
y_pred = df['out_dectlist'].to_numpy()  
y_true = df['out_dectlabel'].to_numpy()
label_name = ["normal","fault"]
        # outputslist outputslabels

draw_confusion_matrix(y_true,y_pred,label_name=label_name,
                      pdf_save_path="decthunxiao.pdf")