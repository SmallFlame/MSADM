# -*-coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import csv
import pandas as pd
import numpy as np
import argparse

nNode = 31

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu") 
# Hyper Parameters
num_e2e = 3
input_size = 18  # 输入数据的维度
length = 96  # 输入数据的长度
num_classes = 7
# 训练集
class TrainData(torch.utils.data.Dataset):
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
        # for line in csv_reader:
        #     row = list(map(float, line))
        #     sample.append(row)
        for _ in range(num_e2e):
            math_data.append([])
        for line in ruler_reader:
            d = line[:36]
            row = list(map(float,d ))
            ruler_data.append(row)
            break
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
        mathf.close()
        rulerf.close()

        math_data = torch.tensor(math_data, dtype=torch.float32).to(device)
        ruler_data = torch.tensor(ruler_data, dtype=torch.float32).to(device)
        label = torch.tensor(label, dtype=torch.int64).to(device)
        blabel = torch.tensor(blabel, dtype=torch.int64).to(device)
        # mean_a = torch.mean(data, dim=1)
        # std_a = torch.std(data, dim=1)

        # # Do Z-score standardization on 2D tensor
        # n_a = data.sub_(mean_a[:, None]).div_(std_a[:, None])

        return math_data,ruler_data, label, blabel, file

# 测试集
class TestData(torch.utils.data.Dataset):
    def __init__(self, math_path,ruler_path):
        self.math_path = math_path+ '/'
        self.ruler_path = ruler_path+ '/'
        self.datas = {}
        self.file_label = {}
        label = 0
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
                        seed = file.split("_")[0] + file.split("_")[1]
                        if seed in self.datas:
                            self.datas[seed].append((file, label, blabel))
                        else:
                            self.datas[seed] = [(file, label, blabel)]
                self.file_label[label] = dir
                label += 1
        # print(self.datas[0])
        i = 0
        for v in self.datas.values():
            if v[0][0].split('_')[0] == 'normal':
                i += 1
        # print(i)
        self.datas = list(self.datas.values())
                
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        item = []
        # print(self.datas[index])
        for file, label, blabel in self.datas[index]:
            f = open(self.data_path + '/' + file.split('_')[0] + '/' + file, 'r+')
            rulerf = open(self.ruler_path + '/' + file.split('_')[0] + '/' + file, 'r+')
            csv_reader = csv.reader(f)
            ruler_reader = csv.reader(rulerf)
            sample = []
            rulers = []
            for line in ruler_reader:
                d = line[:36]
                row = list(map(float,d ))
                rulers.append(row)
                break
            for _ in range(num_e2e):
                sample.append([])
            for line in csv_reader:
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
                sample[0].append(e2e0)
                sample[1].append(e2e1)
                sample[2].append(e2e2)
            f.close()

            sample = torch.tensor(sample, dtype=torch.float32).to(device)
            rulers = torch.tensor(rulers, dtype=torch.float32).to(device)
            label = torch.tensor(label, dtype=torch.int64).to(device)
            blabel = torch.tensor(blabel, dtype=torch.int64).to(device)
            item.append((sample,rulers, label, blabel, file))
        return item

# 端到端注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, 1, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(1, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # x 的输入格式是：[batch_size, C, H, W]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # print(self.avg_pool(x).shape, self.fc1(self.avg_pool(x)).shape, avg_out.shape)
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # keepdim=True:保持维度不变
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # max_out：最大值，_：最大值的索引
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class E2ELSTM(nn.Module):
    def __init__(self):
        # 50 200 0.0002 2 69.31 85.03
        super(E2ELSTM, self).__init__()
        self.num_layers = 2
        self.conv_size = 36
        # self.conv_size = 48        
        self.hidden_size = 64
        self.channel_attention = ChannelAttention(3)
        self.spacial_attention = SpatialAttention()
        # self.linear1 = nn.Linear(input_size, 16)
        self.conv = nn.Sequential(
            nn.Conv2d(num_e2e, 1, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(1),
            )
        self.lstm = nn.LSTM(self.conv_size, self.hidden_size, self.num_layers, batch_first=True)  # batch_first=True仅仅针对输入而言
        self.linear0 = nn.Linear(self.hidden_size+36, int(self.hidden_size / 2))
        self.linear1 = nn.Linear(self.hidden_size, int(self.hidden_size / 2))
        self.linear2 = nn.Linear(int(self.hidden_size / 2), int(self.hidden_size / 4))
        self.linearDiog = nn.Linear(int(self.hidden_size / 4), num_classes)
        self.linearDect = nn.Linear(int(self.hidden_size / 4), 2)
        self.normDiog = nn.BatchNorm1d(1)
        self.normDect = nn.BatchNorm1d(1)

    def forward(self, x,rulers):
        # # 设置初始状态h_0与c_0的状态是初始的状态，一般设置为0，尺寸是,x.size(0)
        x = self.channel_attention(x) * x
        x = self.spacial_attention(x) * x
        new = torch.zeros(x.shape[0], x.shape[2], x.shape[3] * 3).to(device)
        for i in range(x.shape[0]):
            e1 = x[i][0]
            e2 = x[i][1]
            e3 = x[i][2]
            new[i] = torch.cat([e1, e2, e3], 1)
        x = self.conv(x)
        x = x.squeeze(1)
        # new = x
        h0 = Variable(torch.randn(self.num_layers, new.size(0), self.hidden_size)).to(device)
        c0 = Variable(torch.randn(self.num_layers, new.size(0), self.hidden_size)).to(device)
        # # Forward propagate RNN
        out, _ = self.lstm(new, (h0, c0))  # 送入一个初始的x值，作为输入以及(h0, c0)
                                        # out's shape (batch_size, 序列长度, hidden_dim)
        out = out[:, -1, :]                # 中间的序列长度取-1，表示取序列中的最后一个数据，这个数据长度为hidden_dim，
        # rulers = rulers.view(-1,36)                                   # 得到的out的shape为(batch_size, hidden_dim)
        # out = torch.cat((out,rulers),dim=1)
        out = self.linear1(out)
        out = self.linear2(out)             # 经过线性层后，out的shape为(batch_size, n_class)
        out_type = self.linearDiog(out)
        # print(out_type.shape)
        out_type = out_type.view(out_type.shape[0], 1, num_classes)
        out_type = self.normDiog(out_type)
        out_type = out_type.view(out_type.shape[0], num_classes)
        out_anomaly = self.linearDect(out)

        return out_type, out_anomaly
    
def train(math_path,ruler_path,learning_rate,batch_size,epoch_num,model_path,init=True): 
    trainData = TrainData(math_path,ruler_path)
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size, shuffle=True)
    model = E2ELSTM()
    model.to(device)
    # Loss and Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    classAcurList = []
    dectAcurList = []
    # Train the Model
    with open(model_path+'/training_results.txt', 'a+') as f:
        for epoch in range(epoch_num):
            epoch_class_corrects = 0  
            epoch_dect_corrects = 0
            epoch_samples = 0  
            for i, (datas,rulers, labels, blabels, files) in enumerate(trainLoader):
                classOutput, dectOutput = model(datas,rulers)
                classLoss = criterion(classOutput, labels)
                dectLoss = criterion(dectOutput, blabels)
                optimizer.zero_grad()
                loss = classLoss + dectLoss
                loss.backward()
                optimizer.step() # 计算精度  
                _, classPreds = torch.max(classOutput, 1)  # 获取预测类别  
                class_corrects = (classPreds == blabels).sum().item()  
                epoch_class_corrects += class_corrects  
                _, dectPreds = torch.max(dectOutput, 1)  # 获取预测类别  
                dect_corrects = (dectPreds == labels).sum().item()  
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
    testData = TestData(math_path,ruler_path)
    testLoader = torch.utils.data.DataLoader(testData, 1, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    model = E2ELSTM()
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
    for item in testLoader:
        for datas,rulers, labels, blabels, files in item:
            outputs, out_dect = model(datas,rulers)            
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