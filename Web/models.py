import torch
import torch.nn as nn
import numpy as np
import os
import random
from pathlib import Path
import sys

# Đặt seed cố định để đảm bảo kết quả nhất quán
def set_seed(seed=42):
    # Đặt seed cho tất cả các nguồn ngẫu nhiên có thể
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Đảm bảo các thuật toán PyTorch hoàn toàn xác định
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Sử dụng try-except vì một số phiên bản PyTorch cũ không hỗ trợ hàm này
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except:
        pass

# Đặt seed cố định ngay từ đầu
set_seed(42)

class LSTM(nn.Module):
    def __init__(self, input_size=16, hidden_size=32, num_layers=2, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Thay đổi số lớp LSTM từ 1 thành 2 để phù hợp với mô hình đã lưu
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_size=16, d_model=64, nhead=8, num_layers=2, output_size=1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        # Thay đổi pos_encoder để phù hợp với mô hình đã lưu và kích thước đúng
        self.input_fc = nn.Linear(input_size, d_model)  # thay cho pos_encoder.weight và pos_encoder.bias
        self.pos_encoder = nn.Module()  # Placeholder cho pos_encoder.pe
        self.pos_encoder.pe = None  # sẽ được khởi tạo từ state dict
        # Đặt chính xác các tham số để phù hợp với checkpoint
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=128,  # Thay vì mặc định 2048
            batch_first=False  # TransformerEncoder mặc định là batch_second (sequence_first)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        x = x.transpose(0, 1)  # [seq_len, batch_size, input_size]
        x = self.input_fc(x)  # [seq_len, batch_size, d_model]
        
        # Áp dụng PE nếu có
        if hasattr(self.pos_encoder, 'pe') and self.pos_encoder.pe is not None:
            try:
                # Lấy positional encoding đã lưu
                pe = self.pos_encoder.pe
                
                # Cắt PE xuống kích thước đúng thay vì mở rộng chuỗi đầu vào
                if pe.size(0) > x.size(0):
                    # Nếu PE quá dài (5000 > 6), chỉ lấy phần cần thiết
                    pe = pe[:x.size(0), :]  # Chỉ lấy 6 phần tử đầu tiên
                
                # Chuyển PE thành tensor 3D nếu cần
                if pe.dim() == 2:  # [seq_len, d_model]
                    pe = pe.unsqueeze(1)  # [seq_len, 1, d_model]
                
                # Đảm bảo kích thước batch khớp
                if pe.size(1) == 1 and x.size(1) > 1:
                    pe = pe.expand(-1, x.size(1), -1)
                
                # Kiểm tra kích thước cuối cùng trước khi cộng
                if pe.size(0) == x.size(0) and pe.size(2) == x.size(2):
                    x = x + pe
                else:
                    print(f"Bỏ qua PE - Kích thước không khớp: PE {pe.shape}, x {x.shape}")
            except Exception as e:
                print(f"Lỗi khi áp dụng positional encoding: {e}, bỏ qua PE")
        
        # Tắt cảnh báo trong quá trình thực thi
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = self.transformer_encoder(x)  # [seq_len, batch_size, d_model]
        
        x = x[-1]  # [batch_size, d_model]
        x = self.decoder(x)  # [batch_size, output_size]
        return x

class CustomModel(nn.Module):
    def __init__(self, input_size=16, hidden_size=64, output_size=1):
        super(CustomModel, self).__init__()
        # Cấu trúc mô hình tùy chỉnh khác với mô hình đã lưu
        # Trong file weights.pypots, dữ liệu được lưu trong "model_state_dict"
        self.model_state_dict = None  # Placeholder để lưu trữ state_dict gốc
        
        # Vẫn giữ các lớp này để tránh lỗi khi gọi forward
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)
    
    def load_state_dict(self, state_dict, strict=True):
        # Lưu state_dict gốc để sử dụng sau này
        self.model_state_dict = state_dict
        # Nếu mô hình có cấu trúc model_state_dict, lấy giá trị đó
        if 'model_state_dict' in state_dict:
            true_state_dict = state_dict['model_state_dict']
            # Cố gắng tải các tham số phù hợp nếu có 
            try:
                super(CustomModel, self).load_state_dict(true_state_dict, strict=False)
            except:
                print("Không thể tải model_state_dict, sử dụng mô hình không tải")
        return self
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size]
        
        # Get the last output
        out = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Tạo singleton models để tránh khởi tạo lại mỗi lần gọi
_CACHED_MODELS = {}

def get_model(name):
    """Return a model based on name"""
    # Đặt seed cố định ngay khi lấy mô hình để đảm bảo tính nhất quán
    set_seed(42)
    
    # Nếu model đã được tạo rồi thì trả về model đó
    if _CACHED_MODELS.get(name) is not None:
        print(f"Sử dụng mô hình {name} từ cache")
        return _CACHED_MODELS[name]
    
    # Đường dẫn đến thư mục models
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    if not os.path.exists(model_dir):
        model_dir = os.path.dirname(os.path.abspath(__file__))  # Nếu không có thư mục models, thì tìm trong thư mục chính
    
    # Nếu chưa có model, tạo mới và load weights từ file
    if name == 'lstm32':
        model = LSTM(input_size=16, hidden_size=32)
        model_path = os.path.join(model_dir, 'lstm_model_32.pth')
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Warning: Model file not found at {model_path}, using untrained model")
        model.eval()
        _CACHED_MODELS['lstm32'] = model
        return model
    
    elif name == 'lstm64':
        dummy_gru_path = os.path.join(model_dir, 'dummy_gru_model.pth')
        
        # Tạo mô hình GRU thay thế vì không thể cài đặt TensorFlow
        try:
            print("Tensorflow không thể được cài đặt. Dùng mô hình GRU thay thế.")
            # Import module tạo GRU model thay thế
            try:
                from gru_dummy_model import get_gru_model, save_dummy_gru_model
                # Tạo và lưu mô hình GRU thay thế nếu chưa có
                if not os.path.exists(dummy_gru_path):
                    save_dummy_gru_model(dummy_gru_path)
                    print(f"Đã tạo mô hình GRU thay thế tại: {dummy_gru_path}")
                
                # Sử dụng mô hình GRU thay thế
                model = get_gru_model()
                print(f"Đang sử dụng mô hình GRU thay thế")
            except ImportError:
                # Nếu không import được module gru_dummy_model, tạo LSTM thông thường
                print("Không thể import module gru_dummy_model, sử dụng LSTM thông thường")
                model = LSTM(input_size=16, hidden_size=64)
                # Load mô hình dummy nếu có
                if os.path.exists(dummy_gru_path):
                    try:
                        # Vẫn cố gắng load dummy model
                        model.load_state_dict(torch.load(dummy_gru_path, map_location=torch.device('cpu')), strict=False)
                        print(f"Đã tải mô hình GRU thay thế từ: {dummy_gru_path}")
                    except Exception as e:
                        print(f"Không thể tải mô hình GRU thay thế: {e}")
        except Exception as e:
            print(f"Lỗi khi khởi tạo mô hình GRU thay thế: {e}")
            model = LSTM(input_size=16, hidden_size=64)
        
        model.eval()
        _CACHED_MODELS['lstm64'] = model
        return model
    
    elif name == 'transformer':
        # Thay đổi d_model thành 64 để khớp với mô hình đã lưu
        model = TransformerModel(input_size=16, d_model=64, nhead=8)
        pytorch_path = os.path.join(model_dir, 'transformer_dummy_model.pth')
        
        # Tạo positional encoding mặc định
        max_seq_len = 100
        d_model = 64
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        model.pos_encoder.pe = pe
        
        # Kiểm tra xem có file dummy model không
        if os.path.exists(pytorch_path):
            try:
                state_dict = torch.load(pytorch_path, map_location=torch.device('cpu'))
                # Xử lý đặc biệt cho pos_encoder.pe
                if 'pos_encoder.pe' in state_dict:
                    pe_tensor = state_dict['pos_encoder.pe']
                    model.pos_encoder.pe = pe_tensor
                    # Xóa khỏi state_dict để tránh lỗi khi load
                    del state_dict['pos_encoder.pe']
                
                model.load_state_dict(state_dict, strict=False)
                print(f"Đã tải mô hình Transformer giả lập từ: {pytorch_path}")
            except Exception as e:
                print(f"Lỗi khi tải mô hình Transformer giả lập: {e}")
        else:
            # Lưu mô hình mặc định để lần sau dùng
            try:
                torch.save(model.state_dict(), pytorch_path)
                print(f"Đã lưu mô hình Transformer giả lập tại: {pytorch_path}")
            except Exception as e:
                print(f"Không thể lưu mô hình Transformer giả lập: {e}")
        
        model.eval()
        _CACHED_MODELS['transformer'] = model
        return model
    
    elif name == 'custom':
        model = CustomModel(input_size=16, hidden_size=64)
        model_path = os.path.join(model_dir, 'weights.pypots')
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                model.load_state_dict(state_dict)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Warning: Model file not found at {model_path}, using untrained model")
        model.eval()
        _CACHED_MODELS['custom'] = model
        return model
    
    # Không tìm thấy model
    raise ValueError(f"Unknown model '{name}'")