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
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu") 
# Hyper Parameters
num_e2e = 3
input_size = 18  # 输入数据的维度
length = 96  # 输入数据的长度
num_classes = 7
batch_size = 40 

# 训练集
class TrainData(torch.utils.data.Dataset):
    def __init__(self, data_path,ruler_path):
        self.data_path = data_path+'/'
        self.ruler_path = ruler_path+'/'
        self.datas = []
        self.file_label = {}
        ilabel = 0
        for _, dirs, _ in os.walk(self.data_path):
            for dir in dirs:
                if dir == 'normal':
                    label = 0
                    blabel = 0
                else:
                    ilabel += 1
                    label = ilabel
                    blabel = 1
                for file in os.listdir(self.data_path + dir):
                    if file[-3:] == 'csv':
                        self.datas.append((file, label, blabel))
                self.file_label[label] = dir              
                
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        file, label, blabel = self.datas[index]
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
        rulerf.close()
        sample = torch.tensor(sample, dtype=torch.float32).to(device)
        rulers = torch.tensor(rulers, dtype=torch.float32).to(device)
        label = torch.tensor(label, dtype=torch.int64).to(device)
        blabel = torch.tensor(blabel, dtype=torch.int64).to(device)
        return sample,rulers, label, blabel, file

# 测试集
class TestData(torch.utils.data.Dataset):
    def __init__(self, data_path,ruler_path):
        self.data_path = data_path + '/'
        self.ruler_path = ruler_path+'/'
        self.datas = {}
        self.file_label = {}
        label = 0
        ilabel = 0
        for _, dirs, _ in os.walk(self.data_path):
            for dir in dirs:
                if dir == 'normal':
                    label = 0
                    blabel = 0
                else:
                    ilabel += 1
                    label = ilabel
                    blabel = 1
                for file in os.listdir(self.data_path + dir):
                    if file[-3:] == 'csv':
                        seed = file.split("_")[1]
                        if seed in self.datas:
                            self.datas[seed].append((file, label, blabel))
                        else:
                            self.datas[seed] = [(file, label, blabel)]
                self.file_label[label] = dir
                label += 1
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

class CNN(nn.Module):
    def __init__(self):
        # 50 200 0.0002 2 69.31 85.03
        super(CNN, self).__init__()
        self.num_layers = 2
        self.conv_size = 64
        # self.conv_size = 48        
        self.hidden_size = 48
        # self.linear1 = nn.Linear(input_size, 16)
        self.conv = nn.Sequential(
            nn.Conv2d(num_e2e, self.conv_size, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(self.conv_size),
            nn.Conv2d(self.conv_size, self.hidden_size, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(self.hidden_size),
            )
        self.linear0 = nn.Linear(72* self.hidden_size+36, int(self.hidden_size / 2))
        self.linear1 = nn.Linear(72* self.hidden_size, int(self.hidden_size / 2))
        self.linear2 = nn.Linear(int(self.hidden_size / 2), int(self.hidden_size / 4))
        self.linearDiog = nn.Linear(int(self.hidden_size / 4), num_classes)
        self.linearDect = nn.Linear(int(self.hidden_size / 4), 2)
        self.normDiog = nn.BatchNorm1d(1)
        self.normDect = nn.BatchNorm1d(1)

    def forward(self, x,ruler):
        x = self.conv(x)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        ruler = ruler.view(-1,36)
        x = torch.cat((x,ruler),dim=1)
        out = self.linear0(x)
        # out = self.linear1(x)
        out = self.linear2(out)
        out_type = self.linearDiog(out)
        out_type = out_type.view(out_type.shape[0], 1, num_classes)
        out_type = self.normDiog(out_type)
        out_type = out_type.view(out_type.shape[0], num_classes)
        out_anomaly = self.linearDect(out)
        out_anomaly = out_anomaly.view(out_anomaly.shape[0], 1, 2)
        out_anomaly = self.normDect(out_anomaly)
        out_anomaly = out_anomaly.view(out_anomaly.shape[0], 2)
        return out_type, out_anomaly

def train(math_path,ruler_path,learning_rate,batch_size,epoch_num,model_path,init=False):
    trainData = TrainData(math_path,ruler_path)
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size, shuffle=True)
    model = CNN()
    model.to(device)
    # Loss and Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # Train the Model
    total_step = len(trainLoader)
    with open(model_path+'/training_results.txt', 'a+') as f:
        for epoch in range(epoch_num):
            epoch_class_corrects = 0  
            epoch_dect_corrects = 0
            epoch_samples = 0  
            for i, (datas,ruler, labels, blabels, files) in enumerate(trainLoader):
                out_diog, out_dect = model(datas,ruler)
                classLoss = criterion(out_diog, labels)
                dectLoss = criterion(out_dect, blabels)
                optimizer.zero_grad()
                loss = classLoss + dectLoss
                loss.backward()
                optimizer.step()
                _, classPreds = torch.max(out_diog, 1)  # 获取预测类别  
                class_corrects = (classPreds == labels).sum().item()  
                epoch_class_corrects += class_corrects  
                _, dectPreds = torch.max(out_dect, 1)  # 获取预测类别  
                dect_corrects = (dectPreds == blabels).sum().item()  
                epoch_dect_corrects += dect_corrects  
                epoch_samples += labels.size(0)
            class_accuracy = 100.0 * epoch_class_corrects / epoch_samples  
            # 如果你也需要计算dectOutput的精度  
            dect_accuracy = 100.0 * epoch_dect_corrects / epoch_samples  
            f.write(f'{epoch + 1},{classLoss:.4f},{dectLoss:.4f},{loss:.4f}, {class_accuracy:.2f}, {dect_accuracy:.2f}\n')  
            torch.save(model.state_dict(), f'{model_path}/model{epoch+1}.pkl')
            print(f'{epoch+1},{classLoss.item()},{dectLoss.item()},{class_accuracy},{dect_accuracy}')

def test(math_path,ruler_path,result_path,model_path):
    testData = TestData(math_path,ruler_path)
    testLoader = torch.utils.data.DataLoader(testData, 1, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    model = CNN()
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
    for item in testLoader:
        for datas,rulers, labels, blabels, files in item:
            outputs, out_dect = model(datas,rulers)    
            probabilities = torch.sigmoid(out_dect[:, 1])              
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