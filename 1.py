import numpy as np  
  
# data = np.random.rand(96)   
  
# data = self.data  
  
num_groups = 8  
group_size = len(data) // num_groups  
groups = [data[i:i+group_size] for i in range(0, len(data), group_size)]  
  
averages = [np.mean(group) for group in groups]  
  
def find_extrema(averages):  
    extrema_count = 0  
    prev_avg = averages[0] if averages else None  
    for avg in averages[1:-1]: 
        if (prev_avg is not None and avg > prev_avg and avg > averages[averages.index(avg)+1]) or  (prev_avg is not None and avg < prev_avg and avg < averages[averages.index(avg)+1]):  
            extrema_count += 1  
        prev_avg = avg  
      
    if averages and len(averages) > 1 and (averages[-1] > averages[-2] or averages[-1] < averages[-2]):  
        extrema_count += 1  
      
    if extrema_count == 0:  
        return 0  
    elif extrema_count == 1 and averages.index(min(averages)) == averages.index(extremes[0]):
        return 1  
    elif extrema_count == 1 and averages.index(max(averages)) == averages.index(extremes[0]):
        return 2  
    else:
        return 3  
  
extremes = find_extrema(averages)  
print(extremes)
