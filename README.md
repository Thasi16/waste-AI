# Waste Classification AI with MobileNetV2 (QAT) 🚮🤖

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)

Hệ thống phân loại rác thải thông minh sử dụng kiến trúc **MobileNetV2** tích hợp kỹ thuật lượng tử hóa **Quantization-Aware Training (QAT)** để tối ưu hóa hiệu năng cực cao trên các thiết bị di động và nhúng.

## 🌟 Tổng quan dự án

Dự án triển khai mô hình MobileNetV2 để phân loại rác thải thành 5 nhóm chính: `bag_other`, `metal`, `mixed`, `paper`, và `plastic`. 

Điểm nổi bật của dự án là việc áp dụng **QAT**, giúp mô hình giữ được độ chính xác gần như tương đương phiên bản gốc nhưng kích thước file giảm đi 4 lần và tốc độ suy luận nhanh hơn đáng kể trên CPU.

## 🛠️ Chi tiết kỹ thuật (Technical Details)

### 1. Kiến trúc MobileNetV2
Mô hình sử dụng **MobileNetV2**, một kiến trúc mạng nơ-ron hiệu quả được tối ưu cho các thiết bị di động với các đặc điểm chính:
*   **Depthwise Separable Convolutions:** Chia phép toán tích chập thông thường thành hai bước (Depthwise và Pointwise), giúp giảm đáng kể số lượng tham số và khối lượng tính toán.
*   **Inverted Residuals:** Sử dụng các kết nối tắt (shortcut) giữa các lớp bottleneck hẹp, giúp truyền thông tin hiệu quả hơn.
*   **Linear Bottlenecks:** Loại bỏ các hàm kích hoạt phi tuyến ở các lớp bottleneck để tránh làm mất thông tin quan trọng.

### 2. Chiến lược huấn luyện (Training Strategy)
Quá trình huấn luyện được thực hiện qua 3 giai đoạn (Phase):
*   **Phase 1 - Feature Extraction:** Đóng băng các lớp của mô hình MobileNetV2 pre-train, chỉ huấn luyện các lớp phân loại mới.
*   **Phase 2 - Fine-tuning:** Mở băng một phần của mô hình gốc để điều chỉnh trọng số sao cho phù hợp nhất với dữ liệu rác thải.
*   **Phase 3 - Quantization-Aware Training (QAT):** Đưa các phép toán lượng tử hóa "giả" vào đồ thị để mô phỏng sự sai số khi chuyển từ Float32 sang INT8, giúp mô hình thích nghi trước khi chuyển đổi chính thức.

### 3. Tối ưu hóa TFLite
*   **Kích thước:** Giảm từ ~10MB (Float32) xuống còn **~2.5MB** (INT8).
*   **Inference:** Tối ưu hóa cực tốt cho CPU điện thoại thông qua các tập lệnh SIMD.

## 📊 Kết quả đánh giá (Results)

Sau quá trình huấn luyện và đánh giá trên tập Test (233 ảnh), mô hình phân loại rác thải **MobileNetV2 (QAT)** đạt được kết quả vô cùng ấn tượng:

### So sánh các phiên bản lượng tử hóa (TFLite)
| Phiên bản | Định dạng | Độ chính xác (Accuracy) | Tốc độ CPU (ms/ảnh) |
| :--- | :--- | :--- | :--- |
| **FP16** | `.tflite` | 85.41% | 13.5 ms |
| **INT8 (QAT)** | `.tflite` | **86.27%** | **16.8 ms** |

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
![Confusion Matrix QAT](https://i.postimg.cc/DwdLCYMW/confusion.png)

*(Quan sát từ Ma trận nhầm lẫn, mô hình phân loại cực kỳ chính xác ở các lớp `paper`, `metal` và `plastic`. Lớp `mixed` (rác hỗn hợp) có tỷ lệ nhầm lẫn cao nhất do đặc tính dữ liệu phức tạp của nó).*

### Demo dự đoán (Inference Demo)
![Inference Demo](https://i.postimg.cc/3RCRXdrS/image.png)




