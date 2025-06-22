import numpy as np
import os
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Kiểm tra xem TensorFlow đã được cài đặt chưa và thử import
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow không được cài đặt. Vui lòng cài đặt bằng 'pip install tensorflow'")
    TF_AVAILABLE = False

def load_keras_model(keras_path):
    """
    Tải mô hình từ file Keras (.keras hoặc .h5)
    """
    if not TF_AVAILABLE:
        print("TensorFlow không được cài đặt. Không thể tải mô hình Keras.")
        print("Hãy cài đặt TensorFlow bằng lệnh: pip install tensorflow")
        return None
    
    try:
        model = tf.keras.models.load_model(keras_path)
        print(f"Đã tải mô hình Keras từ {keras_path}")
        return model
    except Exception as e:
        print(f"Lỗi khi tải mô hình Keras: {e}")
        return None

def keras_to_pytorch(keras_model, pytorch_model):
    """
    Chuyển đổi trọng số từ mô hình Keras sang mô hình PyTorch
    """
    try:
        # Lấy trọng số từ mô hình Keras
        keras_weights = keras_model.get_weights()
        print(f"Đã lấy được trọng số từ Keras: {len(keras_weights)} lớp")
        
        # In thêm thông tin để debug
        for i, w in enumerate(keras_weights):
            print(f"Layer {i}: shape = {w.shape}")
        
        # Kiểm tra kiểu mô hình PyTorch
        if isinstance(pytorch_model, nn.LSTM) or 'LSTM' in pytorch_model.__class__.__name__:
            print(f"Nhận dạng là mô hình LSTM, đang chuyển đổi...")
            return keras_lstm_to_pytorch_lstm(keras_weights, pytorch_model)
        elif 'Transformer' in pytorch_model.__class__.__name__:
            print(f"Nhận dạng là mô hình Transformer, đang chuyển đổi...")
            return keras_transformer_to_pytorch_transformer(keras_weights, pytorch_model)
        else:
            print(f"Không hỗ trợ chuyển đổi cho loại mô hình: {pytorch_model.__class__.__name__}")
            return False
    except Exception as e:
        print(f"Lỗi khi chuyển đổi trọng số: {e}")
        import traceback
        traceback.print_exc()
        return False

def keras_lstm_to_pytorch_lstm(keras_weights, pytorch_lstm):
    """
    Chuyển đổi trọng số LSTM từ Keras sang PyTorch
    """
    try:
        # Trong Keras, LSTM weights thường có dạng [Wi, Wf, Wc, Wo], [Ui, Uf, Uc, Uo], [bi, bf, bc, bo]
        # Trong PyTorch, LSTM weights có dạng [Wih, Whh, bih, bhh]
        
        # Kiểm tra số lớp và chiều của trọng số
        num_layers = pytorch_lstm.num_layers
        hidden_size = pytorch_lstm.hidden_size
        
        # Cho mạng LSTM cơ bản với 1 lớp
        if num_layers == 1 and len(keras_weights) >= 3:
            # Lớp LSTM đầu tiên
            keras_Wi = keras_weights[0]  # Input weights
            keras_Ui = keras_weights[1]  # Recurrent weights
            keras_bi = keras_weights[2]  # Biases
            
            # Tạo state dict PyTorch
            state_dict = {}
            
            # Chuyển đổi weights input gate
            state_dict['lstm.weight_ih_l0'] = torch.FloatTensor(np.concatenate([keras_Wi]))
            state_dict['lstm.weight_hh_l0'] = torch.FloatTensor(np.concatenate([keras_Ui]))
            state_dict['lstm.bias_ih_l0'] = torch.FloatTensor(keras_bi[:hidden_size*4])
            state_dict['lstm.bias_hh_l0'] = torch.zeros(hidden_size*4)
            
            # Thêm lớp fully connected
            if len(keras_weights) >= 5:
                state_dict['fc.weight'] = torch.FloatTensor(keras_weights[3].T)
                state_dict['fc.bias'] = torch.FloatTensor(keras_weights[4])
            
            # Tải trọng số vào mô hình PyTorch
            pytorch_lstm.load_state_dict(state_dict, strict=False)
            return True
        
        # Cho mạng LSTM nhiều lớp
        elif num_layers > 1 and len(keras_weights) >= num_layers * 3:
            # Tạo state dict PyTorch
            state_dict = {}
            
            for i in range(num_layers):
                keras_Wi_idx = i * 3
                keras_Ui_idx = i * 3 + 1
                keras_bi_idx = i * 3 + 2
                
                if keras_Wi_idx < len(keras_weights) and keras_Ui_idx < len(keras_weights) and keras_bi_idx < len(keras_weights):
                    keras_Wi = keras_weights[keras_Wi_idx]  # Input weights
                    keras_Ui = keras_weights[keras_Ui_idx]  # Recurrent weights
                    keras_bi = keras_weights[keras_bi_idx]  # Biases
                    
                    # Chuyển đổi weights input gate
                    state_dict[f'lstm.weight_ih_l{i}'] = torch.FloatTensor(np.concatenate([keras_Wi]))
                    state_dict[f'lstm.weight_hh_l{i}'] = torch.FloatTensor(np.concatenate([keras_Ui]))
                    state_dict[f'lstm.bias_ih_l{i}'] = torch.FloatTensor(keras_bi[:hidden_size*4])
                    state_dict[f'lstm.bias_hh_l{i}'] = torch.zeros(hidden_size*4)
            
            # Thêm lớp fully connected
            fc_weights_idx = num_layers * 3
            if fc_weights_idx < len(keras_weights) and fc_weights_idx + 1 < len(keras_weights):
                state_dict['fc.weight'] = torch.FloatTensor(keras_weights[fc_weights_idx].T)
                state_dict['fc.bias'] = torch.FloatTensor(keras_weights[fc_weights_idx + 1])
            
            # Tải trọng số vào mô hình PyTorch
            pytorch_lstm.load_state_dict(state_dict, strict=False)
            return True
        
        else:
            print("Cấu trúc trọng số Keras không tương thích với mô hình PyTorch LSTM")
            return False
            
    except Exception as e:
        print(f"Lỗi khi chuyển đổi trọng số LSTM: {e}")
        return False

def keras_transformer_to_pytorch_transformer(keras_weights, pytorch_transformer):
    """
    Chuyển đổi trọng số Transformer từ Keras sang PyTorch
    """
    try:
        # Phân tích cấu trúc Transformer là phức tạp và phụ thuộc vào cách cài đặt
        # Cần phân tích chi tiết cấu trúc của cả hai mô hình
        print("Đang chuyển đổi mô hình Transformer... (chức năng này có thể không hoàn hảo)")
        
        # Tạo state dict cơ bản cho Transformer
        state_dict = {}
        
        # Xác định các thành phần cơ bản
        d_model = pytorch_transformer.input_fc.out_features
        
        # Trong nhiều trường hợp, chúng ta cần ánh xạ thủ công 
        # từng tham số từ mô hình Keras sang mô hình PyTorch
        
        # Ví dụ: chuyển đổi các lớp input và output
        if len(keras_weights) >= 2:  # Giả định 2 lớp đầu tiên là input embedding
            state_dict['input_fc.weight'] = torch.FloatTensor(keras_weights[0].T)
            state_dict['input_fc.bias'] = torch.FloatTensor(keras_weights[1])
        
        if len(keras_weights) >= 4:  # Giả định 2 lớp cuối là output layer
            state_dict['decoder.weight'] = torch.FloatTensor(keras_weights[-2].T)
            state_dict['decoder.bias'] = torch.FloatTensor(keras_weights[-1])
        
        # Tạo positional encoding đơn giản
        max_seq_len = 100
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        state_dict['pos_encoder.pe'] = pe
        
        # Tải trọng số vào mô hình PyTorch
        pytorch_transformer.pos_encoder.pe = pe
        pytorch_transformer.load_state_dict(state_dict, strict=False)
        
        print("Đã chuyển đổi thành công một số tham số của mô hình Transformer")
        return True
        
    except Exception as e:
        print(f"Lỗi khi chuyển đổi trọng số Transformer: {e}")
        return False

def convert_keras_to_pytorch(keras_path, pytorch_model, save_path=None):
    """
    Chuyển đổi mô hình Keras sang định dạng PyTorch và tùy chọn lưu lại
    
    Args:
        keras_path (str): Đường dẫn đến file mô hình Keras (.keras hoặc .h5)
        pytorch_model (nn.Module): Mô hình PyTorch cần tải trọng số
        save_path (str, optional): Đường dẫn để lưu mô hình PyTorch. Mặc định là None.
        
    Returns:
        bool: True nếu chuyển đổi thành công, False nếu thất bại
    """
    # Kiểm tra TensorFlow có sẵn không
    if not TF_AVAILABLE:
        print("Không thể chuyển đổi: TensorFlow không được cài đặt")
        print("Vui lòng cài đặt TensorFlow: pip install tensorflow")
        print("Sau đó khởi động lại ứng dụng")
        return False
    
    # Tải mô hình Keras
    keras_model = load_keras_model(keras_path)
    if keras_model is None:
        return False
    
    # Chuyển đổi trọng số từ Keras sang PyTorch
    success = keras_to_pytorch(keras_model, pytorch_model)
    
    if success and save_path:
        # Lưu mô hình PyTorch
        torch.save(pytorch_model.state_dict(), save_path)
        print(f"Đã lưu mô hình PyTorch vào {save_path}")
    
    return success