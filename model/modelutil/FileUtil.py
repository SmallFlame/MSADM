import pandas as pd
import csv  
  
def read_ruler(read_path):
    # 打开CSV文件  
    print(f"读取{read_path}中文件......")
    res = []
    with open(read_path, 'r') as csvfile:  
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            res.append(row)
    return res

def read_by_csv(read_path):
    df = pd.read_csv(read_path, header=None)
    print(f"读取{read_path}中文件......")
    return df

def write_by_csv(save_path, data):
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False, header=False)
    print(f"保存文件在{save_path}......")
