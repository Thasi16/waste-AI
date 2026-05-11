# Dự án Phân loại Rác thải với MobileNetV2 (QAT - Quantization-Aware Training)

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
Dự án này triển khai mô hình **MobileNetV2** để phân loại rác thải thành 5 nhóm: `bag_other`, `metal`, `mixed`, `paper`, và `plastic`. Đặc biệt, dự án hỗ trợ tính năng **Quantization-Aware Training (QAT)** để tạo ra các mô hình TFLite siêu nhẹ (INT8), tối ưu hóa tốc độ và dung lượng khi chạy trên các thiết bị nhúng (Edge Devices) hoặc điện thoại di động mà không làm giảm đáng kể độ chính xác.

## ⚙️ Cài đặt môi trường (Prerequisites & Installation)

1. **Clone repository:**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name/mobilenetV2
```

2. **Cài đặt các thư viện cần thiết:**
Sử dụng file `requirements.txt` để thiết lập môi trường:
```bash
pip install -r requirements.txt
```

## 📂 Cấu trúc thư mục

- `config.py`: File cấu hình chứa các tham số siêu biến (hyperparameters) như thư mục lưu file, kích thước ảnh, batch size, số lượng epochs.
- `data_loader.py`: Xử lý việc đọc ảnh, chuẩn bị bộ dữ liệu tf.data.Dataset và tăng cường dữ liệu (Data Augmentation).
- `model.py`: Chứa mã nguồn khởi tạo kiến trúc MobileNetV2 (Feature Extraction & Fine-tuning).
- `train.py`: Tập lệnh chạy huấn luyện mô hình cơ bản trên máy tính cục bộ.
- `predict.py`: File dùng để đánh giá mô hình và dự đoán ảnh thực tế bằng cách sử dụng TFLite models.
- `generate_kaggle_notebook.py`: Script dùng để tự động tạo ra các Notebook Kaggle phục vụ cho việc huấn luyện trên nền tảng đám mây.
- `kaggle_train_mobilenetv2_QAT.ipynb`: Notebook huấn luyện mô hình hoàn chỉnh **bao gồm giai đoạn QAT** (Tự động sinh ra bởi script trên).
- `kaggle_train_mobilenetv2.ipynb`: Notebook huấn luyện mô hình thông thường (Post-Training Quantization).

## 🚀 Hướng dẫn huấn luyện QAT trên Kaggle (Đề xuất)

Do yêu cầu sức mạnh tính toán lớn và sự ổn định khi chạy QAT, khuyến nghị bạn nên chạy tệp Notebook đã được sinh sẵn trên Kaggle.

1. Đăng nhập vào Kaggle và tạo một Notebook mới.
2. Tải Dataset rác thải của bạn lên Notebook (ví dụ: `smart-waste-dataset`).
3. Chọn **File > Import Notebook** và tải tệp `kaggle_train_mobilenetv2_QAT.ipynb` lên.
4. Bật chế độ GPU (Nên dùng GPU P100 hoặc T4).
5. Kiểm tra và chỉnh sửa đường dẫn dữ liệu `DATA_DIR` trong Notebook (nếu cần thiết).
6. Nhấn **Run All**. Notebook sẽ tự động chạy qua 3 Phase:
   - **Phase 1:** Feature Extraction (Đóng băng base_model).
   - **Phase 2:** Fine-tuning (Mở băng một phần base_model).
   - **Phase 3:** QAT (Mô phỏng Quantization trong quá trình học để chuẩn bị chuyển đổi sang INT8).
7. Các file được tạo ra sau khi kết thúc:
   - `mobilenetv2_waste_best.keras`: Mô hình gốc (Float32).
   - `mobilenetv2_waste_qat.h5`: Mô hình đã qua huấn luyện QAT.
   - `mobilenetv2_waste_FP16.tflite`: Mô hình tối ưu Float16.
   - `mobilenetv2_waste_INT8.tflite`: Mô hình nhẹ nhất chạy bằng INT8 (tối ưu CPU/Nhúng).

## 💻 Hướng dẫn chạy cục bộ (Local)

Nếu máy tính của bạn có card đồ họa mạnh, bạn có thể chạy trực tiếp:

**1. Cấu hình**
Vào file `config.py` và sửa đường dẫn `DATA_DIR` trỏ tới thư mục chứa dataset.

**2. Huấn luyện (Không QAT)**
Chạy lệnh sau để train mô hình cơ bản (PTQ):
```bash
python train.py
```
*Lưu ý: `train.py` thực hiện Post-Training Quantization (PTQ) thay vì QAT. Nếu muốn QAT, hãy dùng notebook trên Kaggle.*

**3. Đánh giá và Dự đoán**
Bạn có thể sử dụng các file tflite vừa tạo được (lấy từ Kaggle tải về hoặc train cục bộ) để chạy thử:
```bash
python predict.py
```
File này sẽ load ảnh từ tập Test, dự đoán phân loại bằng file `_INT8.tflite` và hiển thị trực quan hóa ma trận nhầm lẫn (Confusion Matrix).

**Hình ảnh Demo Dự đoán (Inference Demo):**

![Inference Demo](https://i.postimg.cc/QtxVfF5F/demo-prediction.png)
*(Hình ảnh minh họa: Mô hình tự động cắt bóc tách rác thải trong khung hình và hiển thị dự đoán nhãn cùng với độ tự tin - Confidence)*

## 📌 Lưu ý về Quantization
- **Post-Training Quantization (PTQ)**: Biến đổi mô hình từ Float32 sang INT8 sau khi train xong. Phù hợp và nhanh gọn, nhưng đôi lúc giảm Accuracy rõ rệt.
- **Quantization-Aware Training (QAT)**: Mô phỏng nhiễu số học lượng tử (quantization noise) trong lúc train. Cách này giúp mô hình "quen" với sự mất mát dữ liệu, giữ vững độ chính xác tương đương phiên bản Float32.

---

## 📊 Kết quả đánh giá (Results)

Sau quá trình huấn luyện và đánh giá trên tập Test (233 ảnh), mô hình phân loại rác thải **MobileNetV2 (QAT)** đạt được kết quả vô cùng ấn tượng:

### So sánh các phiên bản lượng tử hóa (TFLite)
| Phiên bản | Định dạng | Độ chính xác (Accuracy) | Tốc độ CPU (ms/ảnh) |
| :--- | :--- | :--- | :--- |
| **FP16** | `.tflite` | 85.41% | 13.5 ms |
| **INT8 (QAT)** | `.tflite` | **86.27%** | 16.8 ms |

*Nhận xét: Phiên bản INT8 trải qua quá trình QAT không những bảo toàn được độ chính xác mà thậm chí còn nhỉnh hơn một chút so với bản FP16 (đạt 86.27%), chứng minh tính hiệu quả vượt trội của phương pháp này khi chuẩn bị mô hình cho thiết bị nhúng.*

### Báo cáo phân loại (Classification Report)
```text
              precision    recall  f1-score   support

   bag_other       0.76      0.91      0.83        46
       metal       0.95      0.85      0.90        46
       mixed       0.68      0.69      0.68        39
       paper       0.97      0.95      0.96        61
     plastic       0.92      0.83      0.87        41

    accuracy                           0.86       233
   macro avg       0.86      0.85      0.85       233
weighted avg       0.87      0.86      0.86       233
```

### Ma trận nhầm lẫn (Confusion Matrix)
![Confusion Matrix QAT](image/confusion.png)

*(Quan sát từ Ma trận nhầm lẫn, mô hình phân loại cực kỳ chính xác ở các lớp `paper`, `metal` và `plastic`. Lớp `mixed` (rác hỗn hợp) có tỷ lệ nhầm lẫn cao nhất do đặc tính dữ liệu phức tạp của nó).*
