import torch  
import torch.nn as nn  
class TransformerModel(nn.Module):
    def __init__(self,output_size, input_size=22, num_heads=2, hidden_size=128,  dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        self.attention = nn.MultiheadAttention(input_size, num_heads)
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(input_size)
        
    def forward(self, x):
        # x.size:[40, 22, 96]
        att_out, _ = self.attention(x, x, x)
        att_out = self.dropout(att_out)
        att_out = self.layer_norm(x + att_out)
        output = self.fc(att_out)
        return output