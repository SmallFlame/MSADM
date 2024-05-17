import pandas as pd
import json
# 3,1,1,2,1,1,0,1,1,0,1,1,1,1,1,1,-1,1,1,-1,1,1,1,1,0,-1,0

class TreeNode:  
    def __init__(self, value):  
        self.value = value  
        self.children = []  
  
    def add_child(self, node):  
        self.children.append(node)  
    
    def generate_sentences(self,xnzb, status,data):
        sentence = ""
        try:  
            status = json.loads(status)  
            data = json.loads(data)  
        except json.JSONDecodeError as e:  
            print(f"解析 JSON 字符串时出错: {e}")  
        else:  
            sentence = "The " + root.children[xnzb].value +" shows"
            for i in range(3):
                sentence = sentence + " " + root.children[xnzb].children[i].children[status[i]].value 
                sentence = sentence + " " +  root.children[xnzb].children[i].value
                if i==0 and (xnzb==0 or xnzb==1 or xnzb==5 or xnzb==6):
                    sentence = sentence + " is " +   str(round(data[i]*100,2)) + "%"
                    sentence = sentence +"," 
                if i==1:
                    sentence = sentence +" and has" 
        return sentence +"." 

root = TreeNode('fault')  
LINKPLR = TreeNode('packet loss rate')  
LINKBRR = TreeNode('bits error rate')
Delay = TreeNode('number of neighboring nodes')  
LT = TreeNode('number of routing table caches')
avg = TreeNode('average value')
avg_num = TreeNode('average num')
jitter = TreeNode('fluctuation')
trends = TreeNode('trend')
zbs = ["normal", "slightly high"," high", "very high", "extremely high","complete loss"]
zbss = ["no jitter", "stable", "normal", "minor","significant", "extremely volatile"]
zbsss = ["up", "down", "jitter","rose sharply and then fell", "fell sharply and then rose"]
nums = ["None", "few", "normal", "many", "seriously over"]
for item in nums:
    a = TreeNode(item)  
    avg_num.add_child(a)
for item in zbs:
    a = TreeNode(item)  
    avg.add_child(a)
for item in zbss:
    a = TreeNode(item)  
    jitter.add_child(a)

for item in zbsss:
    a = TreeNode(item)  
    trends.add_child(a)
index = 0
for item in ['packet loss rate','bits error rate','number of neighboring nodes','number of routing table caches','cache size','packet loss rate','bits error rate','transport delay']:
    w = TreeNode(item)
    if index==0 or index ==1 or index== 5 or index==6:
        w.add_child(avg)
    else:
        w.add_child(avg_num)
    w.add_child(jitter)
    w.add_child(trends)
    root.add_child(w)
    index = index+1

def gentext(rule_path,result,eq_label,result_path):
    # 示例：构建树形结构  
    pf = pd.read_csv(rule_path,header=None,).fillna(0)
    status = pf.iloc[0].tolist()  
    data = pf.iloc[1].tolist()  
    sentences = []
    # 生成句子
    sentence = f"The current {eq_label} status is as follows:"
    sentences.append(sentence)
    for i in range(5):
        sentence =  root.generate_sentences(i,status[i],data[i])  
        sentences.append(sentence)

    sentence ="The information about the communication links of the current node is as follows:"
    sentences.append(sentence)
    for i in range(5,8):
        sentence = root.generate_sentences(i,status[i],data[i])  
        sentences.append(sentence)
    sentence=f"The current node may have a fault for {result}!"
    sentences.append(sentence)
    with open(result_path, 'w', encoding='utf-8') as file:  
        for s in sentences:  
            file.write(s + '\n')  
    print("report save:",result_path)