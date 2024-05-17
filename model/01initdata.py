import csv
import os
import os  
import numpy as np
import pandas as pd  
from modelutil.MathModel import MathModel
from modelutil.RulerModel import RulerModel
from modelutil.FileUtil import write_by_csv
"""
初始化训练数据
"""
type1 = "train9to12"
root_dir = f'data/origin/{type1}/'
math_dir = f'data/train/{type1}/math1/'
ruler_dir = f'data/train/{type1}/ruler1/'  
text_dir = f'data/train/{type1}/text1/'
ruler_json = 'data/train/ruler.json'

for subdir, dirs, files in os.walk(root_dir):  
    print(subdir)
    for file in files:  
        if file.endswith(".csv"):  
            filename = subdir.split('/')[-1]
            math_data = math_dir+filename
            ruler_data = ruler_dir+filename
            text_data = text_dir+filename
            if False == os.path.exists(math_data):
                os.makedirs(math_data, exist_ok=True)
            if False == os.path.exists(ruler_data):
                os.makedirs(ruler_data, exist_ok=True)
            if False == os.path.exists(text_data):
                os.makedirs(text_data, exist_ok=True)
            
            origin_data_path = os.path.join(root_dir,filename, file)
            math_data_path = os.path.join(math_dir,filename, file)
            ruler_data_path = os.path.join(ruler_dir,filename, file)
            text_data_path = os.path.join(text_dir,filename, file)
            model = MathModel(origin_data_path)
            data = model.process_data()
            data = data[:,:40]
            columns_to_delete = [7, 8,11,12,13,14,19,20,23,24,25,26,31,32,35,36,37,38]  # 调整为正确的索引  
            data = np.delete(data, columns_to_delete, axis=1)  
            write_by_csv(math_data_path,data)
            model = RulerModel(math_data_path,ruler_json)
            math = model.process_data()
            status = model.get_status()
            ruler_data = [item for sublist in status for item in sublist]
            with open(ruler_data_path, 'w', newline='', encoding='utf-8') as csvfile:  # 'a'表示追加模式  
                writer = csv.writer(csvfile)  
                writer.writerow(ruler_data)
            text_data = [status,math]
            write_by_csv(text_data_path,text_data)
            print(ruler_data_path)
            print(text_data_path)