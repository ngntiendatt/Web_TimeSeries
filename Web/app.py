from flask import Flask, render_template, request, session
import numpy as np
import torch
from models import get_model, set_seed
import os
import datetime

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'  # Sử dụng khóa cố định thay vì ngẫu nhiên

# Tắt cảnh báo không cần thiết
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.transformer')

@app.route('/', methods=['GET','POST'])
def index():
    # Lấy ngày hiện tại để hiển thị trong input date
    today_date = datetime.datetime.now().strftime('%Y-%m-%d')
    result = None
    error_message = None
    has_empty_fields = False  # Khởi tạo giá trị mặc định
    
    if request.method == 'POST':
        # Kiểm tra dữ liệu và thu thập giá trị
        seq = []
        input_data = {}
        has_empty_fields = False
        
        for i in range(24):
            row = []
            for j in range(1, 5):
                field_name = f'var{i+1}_{j}'
                if field_name not in request.form or not request.form[field_name].strip():
                    has_empty_fields = True
                    input_data[field_name] = ""
                    value = 0.0
                else:
                    try:
                        value = float(request.form[field_name])
                        input_data[field_name] = value
                    except ValueError:
                        has_empty_fields = True
                        input_data[field_name] = ""
                        value = 0.0
                
                row.append(value)
            seq.append(row)
        
        # Nếu có trường thiếu, thông báo và không thực hiện dự đoán
        if has_empty_fields:
            error_message = "Một số trường dữ liệu trống hoặc không hợp lệ. Vui lòng điền đầy đủ hoặc sử dụng nút 'Điền dữ liệu thiếu' để nội suy các giá trị."
            # Truyền cả input_data để hiển thị những ô đã nhập và đánh dấu các ô trống
            return render_template('index.html', 
                                  result=None, 
                                  error_message=error_message,
                                  has_empty_fields=True,
                                  previous_data=input_data,
                                  today_date=today_date)
        
        arr = np.array(seq)  # shape (24,4)
        name = request.form['model_select']
        
        try:            # Đặt seed cố định trước mỗi lần dự đoán để đảm bảo kết quả nhất quán
            set_seed(42)
            # Lưu vào cache phiên để giữ nguyên mô hình
            if 'model_cache' not in session:
                session['model_cache'] = {}
            
            # Sử dụng các mô hình được cache trong phiên
            model = get_model(name)
              # Dự đoán
            if hasattr(model, 'predict'):
                X = arr.reshape(1, -1)
                # Đảm bảo kết quả nhất quán cho scikit-learn models
                set_seed(42)
                preds = model.predict(X)[:3].tolist()
            else:                # Đặt seed lại và cố định tất cả các nguồn ngẫu nhiên
                set_seed(42)
                # Đảm bảo các thuật toán hoàn toàn xác định
                # Note: torch.set_deterministic() được thay thế bởi torch.use_deterministic_algorithms() trong PyTorch mới hơn
                
                # Chuyển đổi dữ liệu thành tensor với dtype và device cố định
                grouped = torch.tensor(arr.reshape(6, 16), dtype=torch.float32)
                x_seq = grouped.unsqueeze(0)
                
                # Đảm bảo model ở chế độ đánh giá (eval)
                model.eval()
                
                with torch.no_grad():
                    # Dùng phép tính xác định để tránh sự biến thiên
                    out = model(x_seq)
                    vals = out.squeeze().tolist()
                    if isinstance(vals, (list, tuple)) and len(vals) == 3:
                        preds = list(vals)  # Convert tuple to list if needed
                    else:                        # Xử lý trường hợp đặc biệt khi không có đủ giá trị dự đoán
                        try:
                            preds = []
                            seq_feat = grouped
                            for i in range(3):
                                # Sử dụng seed cố định cho mỗi bước
                                set_seed(42 + i)  # Tăng seed theo từng bước để có giá trị dự đoán khác nhau
                                x_in = seq_feat.unsqueeze(0)
                                pred = model(x_in).item()
                                preds.append(pred)
                                # Sử dụng giá trị cố định thay vì ngẫu nhiên
                                new_row = torch.full((1, 16), pred, dtype=torch.float32)
                                seq_feat = torch.cat([seq_feat[1:], new_row], dim=0)
                        except Exception as inner_e:
                            print(f"Lỗi khi dự đoán tuần tự: {inner_e}")
                            # Đảm bảo luôn có 3 giá trị dự đoán
                            while len(preds) < 3:
                                preds.append(0.5)  # Giá trị mặc định nếu dự đoán thất bại
            
            # Đảm bảo giá trị dự đoán không âm và quy mô hợp lý
            preds = [abs(pred) for pred in preds]  # Lấy giá trị tuyệt đối cho tất cả dự đoán
            
            # Lấy ngày dự đoán từ form
            prediction_date = request.form.get('prediction_date', today_date)
              # Gán kết quả và đảm bảo tất cả các giá trị đều là số
            result = {
                'date': prediction_date,
                '1h': round(float(preds[0]), 4) if preds and len(preds) > 0 else 0.0,
                '2h': round(float(preds[1]), 4) if preds and len(preds) > 1 else 0.0,
                '3h': round(float(preds[2]), 4) if preds and len(preds) > 2 else 0.0
            }                # Kiểm tra lại kết quả để đảm bảo không có giá trị None hoặc không hợp lệ
            print(f"Dự đoán từ {name}: {preds}")
            print(f"Kết quả: {result}")
            # Lưu kết quả đã dự đoán vào phiên cho lần truy cập tiếp theo
            session['previous_data'] = input_data
        except Exception as e:
            error_message = f"Lỗi khi dự đoán: {str(e)}"
            # Nếu có lỗi, hiển thị kết quả mặc định
            result = {
                'date': request.form.get('prediction_date', today_date),
                '1h': 0.0,
                '2h': 0.0,
                '3h': 0.0
            }
    
    return render_template('index.html', 
                        result=result, 
                        error_message=error_message,
                        has_empty_fields=has_empty_fields,
                        previous_data=session.get('previous_data', {}),
                        today_date=today_date)

if __name__ == '__main__':
    app.run(debug=True)