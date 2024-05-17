import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F

# "cuda:4" if torch.cuda.is_available() else

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
# 超参数定义
batch_size = 20
# num_epochs = 2
input_dim = 22
length = 96
output_dim = 7
# transformer层数
num_layers = 2
dropout = 0.3

# 读入数据集
# 同时进行多分类和二分类的标签划分
class FaultDataset(Dataset):
    def __init__(self, data_path,ruler_path):
        self.file_names = []
        self.labels = []
        self.datas = []
        label_list = ['congest', 'malicious', 'out', 'nodedown', 'normal', 'obstacle', 'appdown']
        # 遍历所有文件夹并获取文件名及其标签
        for label in label_list:
            folder_path = os.path.join(data_path, label)
            r_path = os.path.join(ruler_path, label)
            for file in os.listdir(folder_path):
                if file.endswith(".csv"):
                    file_path1 = os.path.join(folder_path, file)
                    file_path2 = os.path.join(r_path, file)
                    if label == 'normal':
                        binary_label = 0
                    else:
                        binary_label = 1
                    self.datas.append((file_path1,file_path2, label_list.index(label), binary_label))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        dir1,dir2, label, binary_label = self.datas[index]
        math = pd.read_csv(dir1, header=None).values.astype(np.float32)
        ruler = pd.read_csv(dir2, header=None).iloc[0,:36].values.astype(np.float32)
        # combined_data = np.vstack([math, ruler])  
        data = torch.tensor(math)
        ruler = torch.tensor(ruler)
        label = torch.tensor(label, dtype=torch.int64)
        binary_label = torch.tensor(binary_label, dtype=torch.float32)
        bn = torch.nn.BatchNorm1d(input_dim, affine=False)
        data = bn(data)
        return data,ruler, label, binary_label


# 搭建CNN+Transformer神经网络
class OneDCNNClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, dropout):
        super(OneDCNNClassifier, self).__init__()
        self.bn = torch.nn.BatchNorm1d(input_dim, affine=False)
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc0 = nn.Linear(128 * length+36, 128 * length)
        self.fc1 = nn.Linear(128 * length, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.fc4 = nn.Linear(128, 2)  # binary classification layer
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=16, dim_feedforward=512, dropout=dropout),
            num_layers=num_layers
        )
# binary_output = torch.sigmoid(self.fc4(x))  # for binary classification
    def forward(self, x,rulers):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # 调整形状
        x = x.permute(2, 0, 1)
        # 使用transformer层
        x = self.transformer(x)
        # 调整形状
        x = x.permute(1, 0, 2)  
        x = x.reshape(x.size(0), -1)
        rulers = rulers.view(-1,36)
        x = torch.cat((x,rulers),dim=1)
        x = self.fc0(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        multiclass_output = self.fc3(x)
        binary_output = self.fc4(x)
        
        return multiclass_output, binary_output

def train(math_path,ruler_path,learning_rate,batch_size,epoch_num,model_path='',init=True):
    train_dataset = FaultDataset(math_path,ruler_path)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    model = OneDCNNClassifier(input_dim, output_dim, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    if init==False:
        model_state_dict = torch.load('out/model/TransformerRE/model20.pkl' , map_location=device)  
        model.load_state_dict(model_state_dict)  
    # 训练测试阶段，使用多任务学习
    with open(model_path+'/training_results.txt', 'a+') as f:
        for epoch in range(epoch_num):
            model.train()
            epoch_class_corrects = 0  
            epoch_dect_corrects = 0
            epoch_samples = 0  
            for i, (datas,rulers, labels, blabels) in enumerate(train_loader):
                classOutput,dectOutput = model(datas,rulers)
                blabels = blabels.long()
                classLoss = criterion(classOutput,labels )
                dectLoss = criterion(dectOutput,blabels )
                optimizer.zero_grad()
                loss = classLoss + dectLoss
                loss.backward()
                optimizer.step()
                # 计算精度  
                _, classPreds = torch.max(classOutput, 1)  # 获取预测类别  
                class_corrects = (classPreds == labels).sum().item()  
                epoch_class_corrects += class_corrects  
                _, dectPreds = torch.max(dectOutput, 1)  # 获取预测类别  
                dect_corrects = (dectPreds == blabels).sum().item()  
                epoch_dect_corrects += dect_corrects  
                epoch_samples += labels.size(0)
            class_accuracy = 100.0 * epoch_class_corrects / epoch_samples  
            dect_accuracy = 100.0 * epoch_dect_corrects / epoch_samples  
            f.write(f'{epoch + 1},{classLoss:.4f},{dectLoss:.4f},{loss:.4f}, {class_accuracy:.2f}, {dect_accuracy:.2f}\n')  
            torch.save(model.state_dict(), f'{model_path}/model{epoch + 1}.pkl')
            print(f'{epoch+1},{classLoss.item()},{dectLoss.item()},{class_accuracy},{dect_accuracy}')

def dect(math_path,ruler_path,model_path):
    # obstacle 1 1
    # out 2 1
    # normal 0 0
    # nodedown 3 1
    # congest 4 1
    # malicious 5 1
    # appdown 6 1
    resC = ["normal","obstacle","out","nodedown","congest","malicious","appdown"]
    resD = ["normal","anomaly"]
    model = OneDCNNClassifier(input_dim, output_dim, num_layers, dropout)
    model.to(device)
    model_state_dict = torch.load(model_path , map_location=device)  
    model.load_state_dict(model_state_dict)  
    model.eval()  

    bn = torch.nn.BatchNorm1d(input_dim, affine=False)
    math = pd.read_csv(math_path, header=None).values.astype(np.float32)
    math.reshape(1,-1)
    data = torch.tensor(math)
    data = bn(data).unsqueeze(0)  
    ruler = pd.read_csv(ruler_path, header=None).iloc[0,:36].values.astype(np.float32)
    ruler = torch.tensor(ruler)
    outputs, out_dect = model(data,ruler)     
    probabilities = torch.sigmoid(out_dect[:, 1])
    _, pred_labels = torch.max(outputs, 1)  
    _, pred_dect = torch.max(out_dect, 1)   
    return resC[pred_labels],resD[pred_dect]
def test(math_path,ruler_path,result_path,model_path):
    testData = FaultDataset(math_path,ruler_path)
    testLoader = torch.utils.data.DataLoader(testData, 1, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    model = OneDCNNClassifier(input_dim, output_dim, num_layers, dropout)
    model.to(device)
    model_state_dict = torch.load(model_path , map_location=device)  
    model.load_state_dict(model_state_dict)  
    model.eval()  
    # 打开或创建csv文件  
    totaltotal = 0
    outputslist = []
    outputslabels = []
    out_dectlist = []
    out_dectlabel = []
    lossC = []
    lossD = []
    depred = []
    delabel = []
    total = 0
    for datas,rulers, labels, blabels  in testLoader:
        outputs, out_dect = model(datas,rulers)     
        blabels = blabels.long()    
        probabilities = torch.sigmoid(out_dect[:, 1])
        depred.extend(probabilities.detach().numpy())
        delabel.extend(blabels.detach().numpy())
        loss1 = criterion(outputs, labels)
        loss2 = criterion(out_dect, blabels)
        _, pred_labels = torch.max(outputs, 1)   
        outputslist.extend(pred_labels.cpu().numpy().tolist())
        outputslabels.extend(labels.cpu().numpy().tolist())
        _, pred_dect = torch.max(out_dect, 1)   
        out_dectlist.extend(pred_dect.cpu().numpy().tolist())
        out_dectlabel.extend(blabels.cpu().numpy().tolist())
        lossC.append(loss1.item())
        lossD.append(loss2.item())
        totaltotal += 1
    df = pd.DataFrame()  
    df['outputslist'] = [o for o in outputslist] 
    df['outputslabels'] =  [out for out in outputslabels]
    df['out_dectlist'] = [d for d in out_dectlist]
    df['out_dectlabel'] = [out for out in out_dectlabel]
    df['lossC'] = lossC
    df['lossD'] = lossD
    df.to_csv(result_path, index=False)  
    print(f"数据已保存到 {result_path} 文件中。")
    # return probabilities,delabel