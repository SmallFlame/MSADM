import os
from RulerModel import RulerModel
import numpy as np
from modelutil.FileUtil import write_by_csv,read_by_csv
from modelutil.DistUtil import get_dist_data
import json
math_path = "project/data/math/normal"
save_save_pathpath = "project/out"
ruler_json = ""
index = 0
def get_bins_counts():
	w = {"ruler_node":
			{"PLR":[[0,0.2,0.4,0.6],[0,0.2,0.4,0.6],[0,0.2,0.4,0.6]],
				"BEN":[[0,0.2,0.4,0.6],[0,0.2,0.4,0.6],[0,0.2,0.4,0.6]],
				"ANN":[[0,0.2,0.4,0.6],[0,0.2,0.4,0.6],[0,0.2,0.4,0.6]],
				"RN":[[0,0.2,0.4,0.6],[0,0.2,0.4,0.6],[0,0.2,0.4,0.6]],
				"CS":[[0,0.2,0.4,0.6],[0,0.2,0.4,0.6],[0,0.2,0.4,0.6]]},
		"ruler_link":{"LinkPLR":[[0,0.2,0.4,0.6],[0,0.2,0.4,0.6],[0,0.2,0.4,0.6]],
								"LinkBEN":[[0,0.2,0.4,0.6],[0,0.2,0.4,0.6],[0,0.2,0.4,0.6]],
								"LinkDelay":[[0,0.2,0.4,0.6],[0,0.2,0.4,0.6],[0,0.2,0.4,0.6]],
								"LinkLT":[[0,0.2,0.4,0.6],[0,0.2,0.4,0.6],[0,0.2,0.4,0.6]]}}
	for item in ["PLR","BEN","ANN","RN","CS"]:
		data = read_by_csv(f"project/out/{item}.csv")
		data1 = get_dist_data(data.iloc[0,:],"plr-avg.png")
		data2 = get_dist_data(data.iloc[1,:],"plr-jitter.png")
		data3 = get_dist_data(data.iloc[2,:],"plr-jitter.png")
		a = []
		a.append(data1)
		a.append(data2)
		a.append(data3)
		w["ruler_node"][item] = a
	for item in ["LinkPLR","LinkBEN","LinkDelay","LinkLT"]:
		data = read_by_csv(f"project/out/{item}.csv")
		data1 = get_dist_data(data.iloc[0,:],"plr-avg.png")
		data2 = get_dist_data(data.iloc[1,:],"plr-jitter.png")
		data3 = get_dist_data(data.iloc[2,:],"plr-jitter.png")
		a = []
		a.append(data1)
		a.append(data2)
		a.append(data3)
		w["ruler_link"][item] = a
	with open("data/train/ruler.json", 'w') as f:  
		json.dump(w,f)
	# print(data.iloc[2,:])
	# get_dist_data(data.iloc[2,:],"plr-van.png")

def save_dist_data(save_save_pathpath):
	for item in ["PLR","BEN","ANN","RN","CS"]:
		avg = []
		jitter = []
		van = []
		index = 0
		for subdir, dirs, files in os.walk(math_path):  
			for file in files:  
				if file.endswith(".csv"):
					math_data_path = os.path.join(math_path, file)
					# print(math_data_path)
					rulerModel = RulerModel(math_data_path,ruler_json)
					data = rulerModel.compute_attribute(index)
					avg.append(data[0])
					jitter.append(data[1])
					van.append(data[2])
		res = [avg,jitter,van]
		index = index+1
		write_by_csv(f"{save_save_pathpath}/{item}.csv",res)
	index = 5
	for item in ["LinkPLR","LinkBRN","LinkDelay","LinkLT"]:
		avg = []
		jitter = []
		van = []
		for subdir, dirs, files in os.walk(math_path):  
			for file in files:  
				if file.endswith(".csv"):
					math_data_path = os.path.join(math_path, file)
					# print(math_data_path)
					rulerModel = RulerModel(math_data_path,ruler_json)
					data1 = rulerModel.compute_attribute(index)
					data2 = rulerModel.compute_attribute(index+6)
					data3 = rulerModel.compute_attribute(index+12)
					avg.append((data1[0]+data2[0]+data3[0])/3)
					jitter.append((data1[1]+data2[1]+data3[1])/3)
					van.append((data1[2]+data2[2]+data3[2])/3)
		res = [avg,jitter,van]
		index = index+1
		write_by_csv(f"{save_save_pathpath}/{item}.csv",res)

if __name__=="__main__":
	# save_dist_data(save_save_pathpath)
	bins = get_bins_counts()