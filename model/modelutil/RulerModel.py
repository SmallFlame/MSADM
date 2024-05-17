import json
import os
import numpy as np
from modelutil.FileUtil import read_by_csv
"""
ruler
"""
class RulerModel:
	"""
	"""
	def __init__(self,math_data,ruler_json,init=False):
			self.data = None
			self.ruler_json = ruler_json
			self.data_path = math_data
			self.filename = self.data_path.split('\\')[-1]
			data = read_by_csv(self.data_path)
			self.data = np.array(data, dtype=float)
			# if init==True:
			# 		self.build_folder()
	
	def build_folder():
		pass
	
	def process_data(self):
		res = []
		for i in range(len(self.data[0])):
			attribute = self.compute_attribute(i)
			res.append(attribute)
		self.math_data = res
		return res
	
	def compute_attribute(self,index):
		avg = self.data[:,index].mean()
		column_data = self.data[:, index]
		diffs = column_data[1:] - column_data[:-1]
		jitter = np.mean(diffs) if diffs.size > 0 else np.nan
		variance = np.var(self.data[:, index]) 
		return [avg,jitter,variance]
	
	def find_extrema(self,averages, H):  
		extrema_count = 0  
		n = len(averages)  
		extrema_points = []
		for i in range(1, n-1): 
			if abs(averages[i] - averages[i-1]) > H and abs(averages[i] - averages[i+1]) > H:  
				if averages[i] > averages[i-1] and averages[i] > averages[i+1]: 
					extrema_count += 1  
					extrema_points.append(i)
				elif averages[i] < averages[i-1] and averages[i] < averages[i+1]: 
					extrema_count += 1  
					extrema_points.append(i)
			if i == 0 and (averages[0] - averages[1] > H or averages[1] - averages[0] > H):  
				if averages[0] > averages[1]:  
					extrema_count += 1  
					extrema_points.append(i)
				elif averages[0] < averages[1]:
					extrema_count += 1  
					extrema_points.append(i)
			if i == n-2 and (averages[-1] - averages[-2] > H or averages[-2] - averages[-1] > H):  
				if averages[-1] > averages[-2]:
					extrema_count += 1  
					extrema_points.append(i)
				elif averages[-1] < averages[-2]:
					extrema_count += 1  
					extrema_points.append(i)
		if extrema_count == 0:  
			return 0  
		elif extrema_count == 1:  
			if averages.index(min(averages)) == extrema_points[0]:  
				return 1  
			elif averages.index(max(averages)) == extrema_points[0]:  
				return 2  
		else:
			return 3  

	def getNodeNum(self,index):
		data = self.data[:,index]
		num_groups = 8  
		group_size = len(data) // num_groups  
		groups = [data[i:i+group_size] for i in range(0, len(data), group_size)] 
		averages = [np.mean(group) for group in groups]  
		extremes = self.find_extrema(averages,0.001)  
		if extremes==None:
			extremes= 0 
		return extremes
	
	def get_status(self):
		with open(self.ruler_json, 'r') as f:  
				ruler = json.load(f)
		rule_dict = {"ruler_node":["PLR","BEN","ANN","RN","CS"],"ruler_link":["PLR","BEN","Delay","LT"]}
		i = 0
		res = []
		for thing in ["ruler_node","ruler_link"]:
			for item in rule_dict[thing]:
				w = 0
				att = []
				for times in ruler[thing][item]:
					status = 0
					for index in range(len(times)-1):
						if self.math_data[i][w] < times[0]:
							att.append(0)
							break
						if self.math_data[i][w] >= times[index] and self.math_data[i][w] <= times[index+1]:
							att.append(status+1)
							break
						if self.math_data[i][w] > times[-1]:
							att.append(-1)
							break
						status = status+1
					w = w+1
				att.append(self.getNodeNum(i))
				res.append(att)
				i=i+1
		return res