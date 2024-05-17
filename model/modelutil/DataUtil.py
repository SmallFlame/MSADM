import os
from modelutil.FileUtil import read_csv
import numpy as np
class DataModel:
    """
    """
    def __init__(self,original_path,init=False):
        self.data = None
        self.data_path = original_path
        self.filename = self.data_path.split('\\')[-1]
        if init==True:
            self.build_folder()
        self.load_data_by_csv()
        self.preprocess()
        # self.get_feature()
    
    def build_folder(self):
        for folder_path in ["feature","normal","code","result"]:
            if not os.path.exists("langue\\out\\"+folder_path):  
                # 如果文件夹不存在，则创建它  
                os.makedirs("langue\\out\\"+folder_path)
    """
    load original data
    """
    def load_data_by_csv(self):
        data = read_csv(self.data_path)
        # num_rows = len(data)
        # num_cols = len(data[0]) if data else 0    
        # if num_rows < 42:    
        #     data += [[0] * num_cols for _ in range(41 - num_rows)]    
        self.data = np.array(data, dtype=float)
    """
    preconditioning data 
        node/link: package loss,byte error
    """
    def preprocess(self):
        self.compute_node_rate(0,8)
        self.compute_node_rate(1,7)
        self.compute_link_rate(5)
        self.compute_link_rate(6)
        return self.data
    """
    compute node loss package/byte rate
    """
    def compute_node_rate(self,node_index,line_index):
        try:
            loss = self.data[:, node_index]  
            accept = self.data[:, line_index] + self.data[:, line_index + 12] + self.data[:, line_index + 24]  
            loss = np.nan_to_num(loss)  
            accept = np.nan_to_num(accept)  
            smoothing_term = 1e-10
            divided = loss / (loss + accept + smoothing_term)
            rounded_divided = np.round(divided * 100) / 100
            self.data[:, node_index] = rounded_divided 
            # self.data = np.delete(self.data, line_index, axis=1) 
        except Exception as e:  
            print(f"An error occurred: {e}")  
            return None
    """
    compute link loss package/byte rate
    """
    def compute_link_rate(self,node_index):
        for i in range(3):
            send = self.data[:,node_index+12*i]+self.data[:,node_index+6+12*i]
            accept = self.data[:,node_index+2+12*i]+self.data[:,node_index+8+12*i]
            send = np.nan_to_num(send)  
            accept = np.nan_to_num(accept)  
            smoothing_term = 1e-10
            # 注意
            lossr = abs(send-accept)/(send+smoothing_term)
            lossr = np.round(lossr * 100) / 100
            self.data[:,node_index+12*i] = lossr
            # self.data = np.delete(self.data, node_index+2+12*i, axis=1) 
            # self.data = np.delete(self.data, node_index+6+12*i, axis=1) 
            # self.data = np.delete(self.data, node_index+8+12*i, axis=1) 
    
    """
    feature analysis
    """
    def get_feature(self):
        normal_dict = {  
            "node":{
                "PLR": [0,0,0,0],  # Package Loss Rate
                "BRN": [0,0,0,0],  # Byte Error Num
                "ANN": [0,0,0,0,0],  # Adjacent Node Num
                "RN": [0,0,0,0,0],  # Route Num
                "CS": [0,0,0,0,0],  # Cache size
            },
            "link":{
                "link1": {
                    "PLR": [0,0,0,0],  # Package Loss Rate
                    "BRN": [0,0,0,0],  # Byte Error Num
                    "LT":  [0,0,0,0],  # Late Time
                },
                "link2": {
                    "PLR": [0,0,0,0],  # Package Loss Rate
                    "BRN": [0,0,0,0],  # Byte Error Num
                    "LT":  [0,0,0,0],  # Late Time
                },
                "link3": {
                    "PLR": [0,0,0,0],  # Package Loss Rate
                    "BRN": [0,0,0,0],  # Byte Error Num
                    "LT":  [0,0,0,0],  # Late Time
                },
            }
        }
        i=0
        for item in ["PLR","BRN","ANN","RN","CS"]:
            normal_dict["node"][item] = self.get_attribute(i)
            i = i + 1
        i=0
        for link in ["link1","link2","link3"]:
            if False==self.is_communicate(i):
                normal_dict["link"][link]["communicate"]="No"
            else:
                normal_dict["link"][link]["communicate"]="Yes"
            w=0
            for item in ["PLR","BRN"]:
                normal_dict["link"][link]["LT"]= self.get_attribute(9+12*i)
                normal_dict["link"][link][item] = self.get_attribute(41+i+w*3)
                w=w+1
            i = i + 1
        
        # with open(Config.baseConfig["out_feature"]+self.filename.split('.')[0]+'.json', 'w+') as f:  
        #     json.dump(normal_dict, f)
        return normal_dict
    """
    get attribute 3 indicator
    """
    def get_attribute(self,index):
        res = []
        # avg
        res.append(self.data[:,index].mean())
        # jitter
        diff_values = np.diff(self.data[:, index])  
        if diff_values.size > 0:  
            mean_diff = np.mean(diff_values)  
        else:  
            mean_diff = 0
        res.append(mean_diff)
        # variance
        res.append(sum([(x - res[0]) ** 2 for x in self.data[:,index]]) / len(self.data[:,index]))
        res.append(0)
        return res
    """
    """
    def is_communicate(self,i):
        res = 0
        for w in range(2):
            a = self.data[:,5+i*12+w].sum()
            b = self.data[:,7+i*12+w].sum()
            c = self.data[:,11+i*12+w].sum()
            d = self.data[:,13+i*12+w].sum()
            if (a+b+c+d)==0:
                res=res+1
        return False if res==2 else True