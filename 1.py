import numpy as np  
  
# 假设data是一个包含96个数据的NumPy数组  
# data = np.random.rand(96)  # 这里可以用随机数据代替你的实际数据  
  
# 假设已经有了data数组  
# data = self.data  
  
# 将数据分为8组，每组12个数据  
num_groups = 8  
group_size = len(data) // num_groups  
groups = [data[i:i+group_size] for i in range(0, len(data), group_size)]  
  
# 计算每组的平均值  
averages = [np.mean(group) for group in groups]  
  
# 判断极值点  
def find_extrema(averages):  
    extrema_count = 0  
    prev_avg = averages[0] if averages else None  
    for avg in averages[1:-1]:  # 忽略第一个和最后一个（因为它们没有两个邻居）  
        if (prev_avg is not None and avg > prev_avg and avg > averages[averages.index(avg)+1]) or  (prev_avg is not None and avg < prev_avg and avg < averages[averages.index(avg)+1]):  
            extrema_count += 1  
        prev_avg = avg  
      
    # 如果最后一个元素是极值（与倒数第二个比较）  
    if averages and len(averages) > 1 and (averages[-1] > averages[-2] or averages[-1] < averages[-2]):  
        extrema_count += 1  
      
    # 根据极值点的类型返回相应的值  
    if extrema_count == 0:  
        return 0  
    elif extrema_count == 1 and averages.index(min(averages)) == averages.index(extremes[0]):  # 只有一个极小值  
        return 1  
    elif extrema_count == 1 and averages.index(max(averages)) == averages.index(extremes[0]):  # 只有一个极大值  
        return 2  
    else:  # 有多个极值点  
        return 3  
  
# 计算极值点个数并返回结果  
extremes = find_extrema(averages)  
print(extremes)