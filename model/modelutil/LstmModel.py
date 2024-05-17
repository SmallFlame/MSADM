# -*-coding: utf-8 -*-
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import os
import csv
import pandas as pd
import numpy as np
import argparse
nNode = 31
# Hyper Parameters
input_size = 18  # 输入数据的维度
length = 96  # 输入数据的长度
num_classes = 7

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 
# 训练集
class TrainData(torch.utils.data.Dataset):
    def __init__(self, math_path,ruler_path):
        self.math_path = math_path
        self.ruler_path = ruler_path
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
                for file in os.listdir(self.math_path + '/' + dir):
                    if file[-3:] == 'csv':
                        self.datas.append((file, label, blabel))
                self.file_label[label] = dir
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        file, label, blabel = self.datas[index]
        f1 = open(self.math_path + '/' + file.split('_')[0] + '/' + file, 'r+')
        f2 = open(self.ruler_path + '/' + file.split('_')[0] + '/' + file, 'r+')
        csv_reader1 = csv.reader(f1)
        csv_reader2 = csv.reader(f2)
        sample = []
        ruler = []
        for line in csv_reader1:
            row1 = list(map(float, line))
            sample.append(row1)
        for line in csv_reader2:
            row2 = list(map(float, line))
            ruler.append(row2)
        f1.close()
        f2.close()
        samples = torch.tensor(sample, dtype=torch.float32).to(device)
        rulers = torch.tensor(ruler, dtype=torch.float32).to(device)
        label = torch.tensor(label, dtype=torch.int64).to(device)
        blabel = torch.tensor(blabel, dtype=torch.int64).to(device)
        
        # mean_a = torch.mean(data, dim=1)
        # std_a = torch.std(data, dim=1)

        # # Do Z-score standardization on 2D tensor
        # n_a = data.sub_(mean_a[:, None]).div_(std_a[:, None])

        return samples,rulers, label, blabel, file

# 测试集
class TestData(torch.utils.data.Dataset):
    def __init__(self,  math_path,ruler_path):
        self.math_path = math_path+'/'
        self.ruler_path = ruler_path+'/'
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
        for file, label, blabel in self.datas[index]:
            f1 = open(self.math_path + '/' + file.split('_')[0] + '/' + file, 'r+')
            f2 = open(self.ruler_path + '/' + file.split('_')[0] + '/' + file, 'r+')
            csv_reader1 = csv.reader(f1)
            csv_reader2 = csv.reader(f2)
            sample = []
            rulers = []
            for line in csv_reader1:
                row1 = list(map(float, line))
                sample.append(row1)
            for line in csv_reader2:
                row2 = list(map(float, line))
                rulers.append(row2)
            f1.close()
            f2.close()
            samples = torch.tensor(sample, dtype=torch.float32).to(device)
            rulers = torch.tensor(rulers, dtype=torch.float32).to(device)
            label = torch.tensor(label, dtype=torch.int64).to(device)
            blabel = torch.tensor(blabel, dtype=torch.int64).to(device)
            item.append((samples,rulers, label, blabel))
        return item

class LSTM(nn.Module):
    def __init__(self):
        # 50 200 0.0002 2 69.31 85.03
        super(LSTM, self).__init__()
        self.num_layers = 2
        self.conv_size = 22
        self.hidden_size = 64
        self.conv = nn.Sequential(
            # num_e2e=3
            nn.Conv2d(3, self.hidden_size, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(1),
            )
        # batch_first=True仅仅针对输入而言pp
        self.lstm = nn.LSTM(self.conv_size, self.hidden_size, self.num_layers, batch_first=True)  
        # self.linear0 = nn.Linear((96*22),96*22)
        self.linear0 = nn.Linear(64+27,int(self.hidden_size/2))
        self.linear1 = nn.Linear(self.hidden_size, int(self.hidden_size / 2))
        self.linear2 = nn.Linear(int(self.hidden_size / 2), int(self.hidden_size / 4))
        self.linearDiog = nn.Linear(int(self.hidden_size / 4), num_classes)
        self.linearDect = nn.Linear(int(self.hidden_size / 4), 2)
        self.normDiog = nn.BatchNorm1d(1)
        self.normDect = nn.BatchNorm1d(1)
        # self.dropout1 = nn.Dropout(p=0.3)
        # self.dropout2 = nn.Dropout(p=0.3)  

    def forward(self, x,ruler): 
        new = x 
        h0 = Variable(torch.randn(self.num_layers, new.size(0), self.hidden_size)).to(device)
        c0 = Variable(torch.randn(self.num_layers, new.size(0), self.hidden_size)).to(device)
        out, _ = self.lstm(new, (h0, c0))  
        out = out[:, -1, :]
        ruler = ruler.view(-1,27)
        out = torch.cat((out, ruler), dim=1) 
        # [40,91]
        out = self.linear0(out)  
        # [40,32]
        # out = self.linear1(out) 
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

def train(math_path,ruler_path,learning_rate,batch_size,epoch_num,model_path='',init=True):
    trainData = TrainData(math_path,ruler_path)
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size, shuffle=True)
    model = LSTM()
    model.to(device)
    if init!=True:
        model_state_dict = torch.load('out/model/LSTMRE/model120.pkl', map_location=device)  
        model.load_state_dict(model_state_dict)  
        print("loading..")
    # Loss and Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    classAcurList = []
    faultAcurList = []
    totalAcurList = []
    start_time = time.time()  # 获取当前时间  
    # Train the Model
    total_step = len(trainLoader)
    print('start training...')
    with open(model_path+'/training_results.txt', 'a+') as f:
        for epoch in range(30):
            loss12 = []
            train_loss = 0
            correct_multiclass = 0
            correct_binary = 0
            total = 0
            for i, (datas,rulers, labels, blabels, files) in enumerate(trainLoader):
                out_diog, out_dect = model(datas,rulers)
                loss1 = criterion(out_diog, labels)
                loss2 = criterion(out_dect, blabels)
                optimizer.zero_grad()
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()
                # 获取多分类任务预测结果的最大概率对应的索引（即预测的类别）  
                _, predicted_multiclass = torch.max(out_diog, 1)  
                # predicted_binary = (out_dect > 0.5).view(-1).long()
                _, predicted_binary = torch.max(out_dect, 1)
                total += labels.size(0)
                correct_multiclass += (predicted_multiclass == labels).sum()
                correct_binary += (predicted_binary == blabels).sum()
            train_loss = train_loss / len(trainLoader)
            train_accuracies_multiclass=100 * correct_multiclass / total
            train_accuracies_binary=100 * correct_binary / total
            f.write(f'{epoch + 121},{loss1:.4f},{loss2:.4f},{loss:.4f}, {train_accuracies_multiclass:.2f}, {train_accuracies_binary:.2f}\n')  
            print(f'{epoch + 121},{loss1:.4f},{loss2:.4f},{loss:.4f}, {train_accuracies_multiclass:.2f}, {train_accuracies_binary:.2f}')
            # torch.save(model.state_dict(),model_path+ f'/model{epoch + 91}.pkl')
    torch.save(model.state_dict(),model_path+ '/model.pkl')

def test(math_path,ruler_path,result_path,model_path):
    testData = TestData(math_path,ruler_path)
    testLoader = torch.utils.data.DataLoader(testData, 1, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    model = LSTM()
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
        for datas,rulers, labels, blabels in item:
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
    #         lossC.append(loss1.item())
    #         lossD.append(loss2.item())
    #         totaltotal += 1
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