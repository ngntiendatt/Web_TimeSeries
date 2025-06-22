import torch
import torch.nn as nn
import numpy as np
import os
import random

# Đặt seed cố định để đảm bảo kết quả nhất quán
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Khởi tạo GRU dummy model - giả lập mô hình GRU khi không có file Keras
class GRUModel(nn.Module):
    def __init__(self, input_size=16, hidden_size=64, num_layers=2, output_size=1):
        super(GRUModel, self).__init__()
        set_seed(42)  # Đảm bảo mô hình giả lập khởi tạo với cùng một seed
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Khởi tạo weights với giá trị cố định
        self._init_weights()
        
    def _init_weights(self):
        set_seed(42)  # Đảm bảo mỗi lần khởi tạo đều giống nhau
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.1)
        
    def forward(self, x):
        # Set initial hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

# Hàm tạo GRU model tương thích với LSTM interface
def get_gru_model():
    model = GRUModel(input_size=16, hidden_size=64)
    # Đặt model ở chế độ eval để đảm bảo tính nhất quán trong dự đoán
    model.eval()
    return model

# Hàm lưu mô hình GRU giả lập
def save_dummy_gru_model(path):
    model = get_gru_model()
    torch.save(model.state_dict(), path)
    return path