from collections import deque
import pandas as pd
import json
from modelutil.MSADM import dect
from modelutil.NLGModel import gentext

class TreeNode:  
    def __init__(self, value):  
        self.value = value  
        self.children = []  
  
    def add_child(self, node):  
        self.children.append(node)  
network1 = TreeNode(0)
network2 = TreeNode(1)
network3 = TreeNode(2)
network4 = TreeNode(3)
network5 = TreeNode(4)
network6 = TreeNode(5)
network2.add_child(network4)
network2.add_child(network5)
network3.add_child(network6)
network1.add_child(network2)
network1.add_child(network3)
# 3,1,1,2,1,1,0,1,1,0,1,1,1,1,1,1,-1,1,1,-1,1,1,1,1,0,-1,0
math_path = ["data/train/train9to12/math1/appdown/appdown_1_0.csv",
            "data/train/train9to12/math1/appdown/appdown_1_2.csv",
            "data/train/train9to12/math1/appdown/appdown_1_4.csv",
            "data/train/train9to12/math1/appdown/appdown_3_6.csv",
            "data/train/train9to12/math1/appdown/appdown_6_8.csv"]
ruler_path = ["data/train/train9to12/ruler1/appdown/appdown_1_0.csv",
            "data/train/train9to12/ruler1/appdown/appdown_1_2.csv",
            "data/train/train9to12/ruler1/appdown/appdown_1_4.csv",
            "data/train/train9to12/ruler1/appdown/appdown_3_6.csv",
            "data/train/train9to12/ruler1/appdown/appdown_6_8.csv"]
text_path = ["data/train/train9to12/text1/appdown/appdown_1_0.csv",
            "data/train/train9to12/text1/appdown/appdown_1_2.csv",
            "data/train/train9to12/text1/appdown/appdown_1_4.csv",
            "data/train/train9to12/text1/appdown/appdown_3_6.csv",
            "data/train/train9to12/text1/appdown/appdown_6_8.csv"]
model_path = "out/model/TransformerRE/model21.pkl"



def collectInformation():  
        queue = deque([network1])  
        while queue:  
            current_node = queue.popleft()
            eq_label = f"node{current_node.value}"
            result_path = f"out/text/{eq_label}.text"
            type,dectN= dect(math_path[current_node.value],ruler_path[current_node.value],model_path)
            print(type)
            if dectN!="normal":
                gentext(text_path[current_node.value],type,eq_label,result_path)
            for child in current_node.children: 
                queue.append(child) 
collectInformation()



