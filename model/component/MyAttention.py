import torch  
import torch.nn as nn  
import torch.nn.functional as F  
# 通道注意力模块  
class ChannelAttention(nn.Module):
    def __init__(self, in_planes=3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, 1, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(1, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # x 的输入格式是：[batch_size, C, H, W]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # print(self.avg_pool(x).shape, self.fc1(self.avg_pool(x)).shape, avg_out.shape)
        out = avg_out + max_out
        return self.sigmoid(out)
# 空间注意力模块
class SpatialAttention(nn.Module):  
    def __init__(self, kernel_size=7):  
        super(SpatialAttention, self).__init__()  
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  
        padding = 3 if kernel_size == 7 else 1  
  
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  
        self.sigmoid = nn.Sigmoid()  
  
    def forward(self, x):  
        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out, _ = torch.max(x, dim=1, keepdim=True)  
        x = torch.cat([avg_out, max_out], dim=1)  
        x = self.conv1(x)  
        return self.sigmoid(x)  
# 时间注意力模块  
class TemporalAttention(nn.Module):  
    def __init__(self, in_features):  
        super(TemporalAttention, self).__init__()  
        self.fc = nn.Linear(in_features, 1)  
        self.softmax = nn.Softmax(dim=1)  
  
    def forward(self, x):  
        print(x.size())
        # x的形状应该是 [batch_size, features, time_steps]  
        e = self.fc(x).squeeze(2)  # [batch_size, time_steps]  
        print(x.size())
        # 计算注意力权重  
        alpha = self.softmax(e) 
        # 应用注意力权重到输入数据  
        return x * alpha.unsqueeze(2)  