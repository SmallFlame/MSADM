import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from component.MyLstm import LSTMModel
from component.MyTransformer import TransformerModel


device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu") 
input_dim = 22

class MyData(Dataset):
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
        data = torch.tensor(math)
        label = torch.tensor(label, dtype=torch.int64)
        binary_label = torch.tensor(binary_label, dtype=torch.float32)
        bn = torch.nn.BatchNorm1d(input_dim, affine=False)
        data = bn(data)
        return data,ruler, label, binary_label
    
class LstmTransformer(nn.Module):
    def __init__(self,output_size,numclasses=7,hidden_size=128):
        # 50 200 0.0002 2 69.31 85.03
        super(LstmTransformer, self).__init__()
        self.transformer = TransformerModel(output_size)
        self.lstm = LSTMModel()
        self.line1 = nn.Linear(hidden_size+36,hidden_size)
        self.line2 = nn.Linear(hidden_size,int(hidden_size/4))
        self.class1 = nn.Linear(int(hidden_size/4),numclasses)
        self.dect = nn.Linear(int(hidden_size/4),2)

    def forward(self,x ,rulers): 
        # x.size():[40, 96, 22]
        # x = x.permute(0, 2, 1)  
        rulers = rulers.view(-1,36)
        # lstm_output.size():40,128
        # x.size():[40, 22, 96]
        transformer_output = self.transformer(x)
        # transformer_output.size():[40, 22, 96]
        lstm_output = self.lstm(transformer_output)
        # lstm_output.size():[40,128]
        combined_output = torch.cat((lstm_output, rulers), dim=1)
        x = self.line1(combined_output)
        x = self.line2(x)
        classT = self.class1(x)
        dectT = self.dect(x)
        return classT,dectT

def train(math_path,ruler_path,learning_rate,batch_size,epoch_num,model_path,init=True):
    trainData = MyData(math_path,ruler_path)
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size, shuffle=True)
    model = LstmTransformer(22) 
    if False==init:
        model_state_dict = torch.load('out/model/LSTMTransformer/model60.pkl' , map_location=device)  
        model.load_state_dict(model_state_dict)  
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    with open(model_path+'/training_results.txt', 'a+') as f:
        for epoch in range(epoch_num): 
            epoch_class_corrects = 0  
            epoch_dect_corrects = 0
            epoch_samples = 0  
            for i, (datas,rulers, labels, blabels) in enumerate(trainLoader):            
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
            torch.save(model.state_dict(), f'{model_path}/model{epoch+1}.pkl')
            print(f'{epoch+61},{classLoss.item()},{dectLoss.item()},{class_accuracy},{dect_accuracy}')


def test(math_path,ruler_path,result_path,model_path):
    testData = MyData(math_path,ruler_path)
    testLoader = torch.utils.data.DataLoader(testData, 1, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    model = LstmTransformer(22)
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
    for datas,rulers, labels, blabels in testLoader:
        blabels = blabels.long()
        outputs, out_dect = model(datas,rulers)       
        probabilities = torch.sigmoid(out_dect[:, 1])
        depred.extend(probabilities.detach().numpy())
        delabel.extend(blabels.detach().numpy()  )
        # loss1 = criterion(outputs, labels)
        # loss2 = criterion(out_dect, blabels)
        # _, pred_labels = torch.max(outputs, 1)   
        # outputslist.extend(pred_labels.cpu().numpy().tolist())
        # outputslabels.extend(labels.cpu().numpy().tolist())
        # _, pred_dect = torch.max(out_dect, 1)   
        # out_dectlist.extend(pred_dect.cpu().numpy().tolist())
        # out_dectlabel.extend(blabels.cpu().numpy().tolist())
    #     lossC.append(loss1.item())
    #     lossD.append(loss2.item())
    #     totaltotal += 1
    # df = pd.DataFrame()  
    # df['outputslist'] = [o for o in outputslist] 
    # df['outputslabels'] =  [out for out in outputslabels]
    # df['out_dectlist'] = [d for d in out_dectlist]
    # df['out_dectlabel'] = [out for out in out_dectlabel]
    # df['lossC'] = lossC
    # df['lossD'] = lossD
    # df.to_csv(result_path, index=False)  
    # print(f"数据已保存到 {result_path} 文件中。")
    return probabilities,delabel