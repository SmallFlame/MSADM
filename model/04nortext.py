


import re

def read_text_file_by_lines(file_path):  
    sentence = []
    word_num= 0
    with open(file_path, 'r', encoding='utf-8') as file:  
        for line in file:  
            num = 0
            for c in line:
                num=num+1
            sentence.append(line)
            # modified_sentence = re.sub(r'([,.%:])', r' \g<1> ', line)  
            # words = modified_sentence.split()  
            # len(words) + len(words) 
            word_num = word_num + num
        return sentence,word_num
# LLM
words_num = 0
sentences = []
sentencess= []
for i in range(3):
    eq_label = f"node{i}"
    text_path = f"out/text/{eq_label}.text"
    sentence,word_num = read_text_file_by_lines(text_path)
    words_num = words_num+word_num
    sentences.append(sentence)
for item in sentences:
    for s in item:
        sentencess.append(s)
# sentencess = [i for arr in sentences for i in arr]
# 8
context = "Current NetWork Context:\n"
# 84
question = "According to the preceding description,if similar historical fault information exists,identify the fault type and provide a solution. If no,identify the current fault type and provide theoptimal solution. Select a fault type from options.\n"
# 78
options = "The fault type mentioned above may not be correct. Determine and confirm the fault according to the Context information. If you have different views on the fault, state the cause.Please select the exception type that best matches Context's performance from the following:a:Node Down; b:Malicious Traffic; c:Network Congestion; d:Communication Obstacles; e:Out-of-Range; f:Network Node Crash."

lwords = ["normal", "slightly high","no jitter", "stable", "None", "few",  "many"]
words_num = words_num + 8+ 84 + 78
dshanchu = [2,3,4,5]
print(words_num)
if words_num>1500:
    for i, ss in enumerate(sentencess[:]):
        modified_sentence = re.sub(r'([,.%:])', r' \g<1> ',ss)  
        wordss = modified_sentence.split()  
        print(lwords)
        for word in lwords:
            print(word)
            print(ss)
            if word in ss:
                words_num = words_num - len(wordss)*2
                dshanchu.append(i)
        if words_num<=1500:
            break

sentencess = [sentencess[i] for i in range(len(sentencess)) if i not in dshanchu]
res = sentencess
res.insert(0,context)
res.append(question)
res.append(options)
result_path= 'finalrepost.txt'
with open(result_path, 'w', encoding='utf-8') as file:  
    for s in res:  
        file.write(s)  
print("report save:",result_path)
        