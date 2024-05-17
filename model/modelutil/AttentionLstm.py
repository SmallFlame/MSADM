import csv
import os

import pandas as pd
from component.MyAttention import ChannelAttention,SpatialAttention
import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu") 
# 数据获取(normal: 0 0 others:1 val>1)
class MyData(torch.utils.data.Dataset):
    def __init__(self, math_path,ruler_path):
        self.math_path = math_path+'/'
        self.ruler_path = ruler_path+'/'
        self.datas = []
        self.file_label = {}
        ilabel = 0
        for _, dirs, _ in os.walk(self.math_path):
            for dir in dirs:
                if dir == 'normal':
                    label = 0
                    blabel = 0
                else:
                    ilabel += 1
                    label = ilabel
                    blabel = 1
                for file in os.listdir(self.math_path + dir):
                    if file[-3:] == 'csv':
                        self.datas.append((file, label, blabel))
                self.file_label[label] = dir              
                
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        file, label, blabel = self.datas[index]
        mathf = open(self.math_path + '/' + file.split('_')[0] + '/' + file, 'r+')
        rulerf = open(self.ruler_path + '/' + file.split('_')[0] + '/' + file, 'r+')
        math_reader = csv.reader(mathf)
        ruler_reader = csv.reader(rulerf)
        math_data = []
        ruler_data = []
        for _ in range(3):
            math_data.append([])
        for line in math_reader:
            row = list(map(float, line))
            e2e0 = row[0:5]
            e2e1 = row[0:5]
            e2e2 = row[0:5]
            for i in range(5, 11):
                e2e0.append(row[i])
            e2e0.append(0)
            for i in range(11, 17):
                e2e1.append(row[i])
            e2e1.append(0)
            for i in range(16, 22):
                e2e2.append(row[i])
            e2e2.append(0)
            math_data[0].append(e2e0)
            math_data[1].append(e2e1)
            math_data[2].append(e2e2)
        
        for line in ruler_reader:
            d = line[:36]
            row = list(map(float,d ))
            ruler_data.append(row)
            break
        
        # print(ruler_data)
        rulerf.close()
        mathf.close()
        math_data = torch.tensor(math_data, dtype=torch.float32).to(device)
        ruler_data = torch.tensor(ruler_data, dtype=torch.float32).to(device)
        label = torch.tensor(label, dtype=torch.int64).to(device)
        blabel = torch.tensor(blabel, dtype=torch.int64).to(device)
        return math_data,ruler_data,label, blabel

# 定义时序预测模型  
class SADMRE(nn.Module):  
    # input_size:输入特征数量;hidden_size:数据质量越低应越高;num_classes:多分类分类数量;
    def __init__(self, input_size=22,lstm_layers=2,hidden_size=64, class_classes=7):  
        super(SADMRE, self).__init__() 
        self.num_layers = lstm_layers
        self.hidden_size = hidden_size
        self.class_classes = class_classes
        self.conv = nn.Sequential(
            nn.Conv2d(3, 1, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(1),
            )
        self.channel_attention = ChannelAttention(3) 
        # self.temporal_attention = TemporalAttention(input_size)  
        self.spacial_attention = SpatialAttention()
        self.lstm = nn.LSTM(6, hidden_size,self.num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size+36,int(hidden_size/2))
        # self.linear2 = nn.Linear(int(hidden_size/2),int(hidden_size/4))
        self.dectLiner = nn.Linear(int(hidden_size/2), 2)  
        self.classLiner = nn.Linear(int(hidden_size/2), class_classes)  
        self.normDiog = nn.BatchNorm1d(1)
        self.normDect = nn.BatchNorm1d(1)
    def forward(self, math,ruler):  
        # 应用时间注意力  
        x = math * self.channel_attention(math)
        x = x*self.spacial_attention(x)  
        new = torch.zeros(x.shape[0], x.shape[2], x.shape[3] * 3).to(device)
        for i in range(x.shape[0]):
            e1 = x[i][0]
            e2 = x[i][1]
            e3 = x[i][2]
            new[i] = torch.cat([e1, e2, e3], 1)
        x = self.conv(x)
        x = x.squeeze(1)
        # 应用LSTM  
        # x = x.cat(ruler)        
        h0 = Variable(torch.randn(self.num_layers, x.size(0), self.hidden_size)).to(device)
        c0 = Variable(torch.randn(self.num_layers, x.size(0), self.hidden_size)).to(device)
        # out, _ = self.lstm(x_attn)  
        lstm_out, _ = self.lstm(x,(h0,c0))  
        # 取最后一个时间步的输出作为预测  
        lstm_out = lstm_out[:, -1, :]  
        ruler = ruler.view(-1,36)
        x_combined = torch.cat((lstm_out, ruler), dim=1)  
        # [40,64]
        # 应用全连接层进行分类或回归  
        out = self.linear1(x_combined)
        # out = self.linear2(out)
        classOutput = self.classLiner(out)  
        dectOutput = self.dectLiner(out)
        
        classOutput = classOutput.view(classOutput.shape[0], 1, 7)
        classOutput = self.normDiog(classOutput)
        classOutput = classOutput.view(classOutput.shape[0], 7)

        dectOutput = dectOutput.view(dectOutput.shape[0], 1, 2)
        dectOutput = self.normDect(dectOutput)
        dectOutput = dectOutput.view(dectOutput.shape[0], 2)
        return classOutput,dectOutput

def train(math_path,ruler_path,learning_rate,batch_size,epoch_num,model_path,init=True):
    trainData = MyData(math_path,ruler_path)
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size, shuffle=True)
    model = SADMRE() 
    if False==init:
        model_state_dict = torch.load(model_path , map_location=device)  
        model.load_state_dict(model_state_dict)  
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    classAcurList = []
    dectAcurList = []
    with open(model_path+'/training_results.txt', 'a+') as f:
        for epoch in range(epoch_num): 
            epoch_class_corrects = 0  
            epoch_dect_corrects = 0
            epoch_samples = 0  
            for i, (datas,rulers, labels, blabels) in enumerate(trainLoader):            
                classOutput,dectOutput = model(datas,rulers)
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
            classAcurList.append(class_accuracy)  
            # 如果你也需要计算dectOutput的精度  
            dect_accuracy = 100.0 * epoch_dect_corrects / epoch_samples  
            dectAcurList.append(dect_accuracy)  
            f.write(f'{epoch + 1},{classLoss:.4f},{dectLoss:.4f},{loss:.4f}, {class_accuracy:.2f}, {dect_accuracy:.2f}\n')  
            torch.save(model.state_dict(), f'{model_path}/model{epoch+1}.pkl')
            print(f'{epoch+1},{classLoss.item()},{dectLoss.item()},{class_accuracy},{dect_accuracy}')

def test(math_path,ruler_path,result_path,model_path):
    testData = MyData(math_path,ruler_path)
    testLoader = torch.utils.data.DataLoader(testData, 1, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    model = SADMRE()
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
        outputs, out_dect = model(datas,rulers)  
        probabilities = torch.sigmoid(out_dect[:, 1])
        depred.extend(probabilities.detach().numpy())
        delabel.extend(blabels.detach().numpy())        
        
    #     loss1 = criterion(outputs, labels)
    #     loss2 = criterion(out_dect, blabels)
    #     _, pred_labels = torch.max(outputs, 1)   
    #     outputslist.extend(pred_labels.cpu().numpy().tolist())
    #     outputslabels.extend(labels.cpu().numpy().tolist())
    #     _, pred_dect = torch.max(out_dect, 1)   
    #     out_dectlist.extend(pred_dect.cpu().numpy().tolist())
    #     out_dectlabel.extend(blabels.cpu().numpy().tolist())
    #     lossC.append(loss1.item())
    #     lossD.append(loss2.item())
        totaltotal += 1
    # df = pd.DataFrame()  
    # df['outputslist'] = [o for o in outputslist] 
    # df['outputslabels'] =  [out for out in outputslabels]
    # df['out_dectlist'] = [d for d in out_dectlist]
    # df['out_dectlabel'] = [out for out in out_dectlabel]
    # df['lossC'] = lossC
    # df['lossD'] = lossD
    # df.to_csv(result_path, index=False)  
    
    # print(f"数据已保存到 {result_path} 文件中。")
    print('all:',totaltotal)
    return probabilities,delabel