from modelutil.FileUtil import read_by_csv
import numpy as np
class MathModel:
	def __init__(self,origin_data) -> None:
		self.origin_data = origin_data
		self.filename = self.origin_data.split('\\')[-1]
		# self.process_data()
		pass

	def process_data(self):
		data = read_by_csv(self.origin_data)
		self.data = np.array(data, dtype=float)
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
					# print(send)
					# print(accept[3])
					send = np.nan_to_num(send)  
					accept = np.nan_to_num(accept)
					lossr = np.zeros_like(send)  
					send_larger = send > accept  
					smoothing_term = 1e-10
					# 当send较大时，计算(send-accept)/send  
					lossr[send_larger] = (send[send_larger] - accept[send_larger]) / (send[send_larger]  +smoothing_term)
					# 当accept较大时，计算(accept-send)/accept  
					lossr[~send_larger] = (accept[~send_larger] - send[~send_larger]) / (accept[~send_larger]  +smoothing_term)
					# lossr = abs(send-accept)/(send+smoothing_term)
					lossr = np.round(lossr * 100) / 100
					# print("lossr",lossr[3])
					self.data[:,node_index+12*i] = lossr