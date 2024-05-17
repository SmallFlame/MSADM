import numpy as np
import matplotlib.pyplot as plt  

def get_dist_data(data,name):
	# mean = np.mean(data)  
	# std_dev = np.std(data)
	total = len(data)
	bins = np.linspace(min(data), max(data), 6) 
	return bins.tolist()
	# 计算每个区间的数据点数量  
	# counts, _ = np.histogram(data, bins=bins)  
	# pinlv = np.round(counts/total,2)
	# plt.bar(bins[:-1], pinlv, width=(bins[1] - bins[0]), edgecolor='black')  
	# plt.title('Distribution of The Average Packet Loss Rate')  
	# plt.xlabel('The Average Packet Loss Rate Range')  
	# plt.ylabel('Frequency of Occurrence')  
	# plt.legend()  
	# plt.savefig(name)