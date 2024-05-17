import torch  
import torch.nn as nn  
class LSTMModel(nn.Module):
    def __init__(self, input_size=22, hidden_size=64, num_layers=2, output_size=128):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x.size():[40,22, 96]
        lstm_out, _ = self.lstm(x)
        # lstm_out.size():[40, 96, 64]
        # lstm_out[:, -1, :].size():[40, 64]
        output = self.fc(lstm_out[:, -1, :])  # 使用最后一个时间步的输出进行分类
        # output.size():[40,128]
        return output