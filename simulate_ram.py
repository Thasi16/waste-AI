import tensorflow as tf
import os
import tracemalloc
import numpy as np
import time

def simulate_ram_usage(tflite_path):
    print(f"\n{'='*50}")
    print(f"ĐANG ĐO LƯỜNG: {tflite_path}")
    print(f"{'='*50}")

    if not os.path.exists(tflite_path):
        print(f"Lỗi: Không tìm thấy file {tflite_path}")
        return

    # Kích thước file trên đĩa cứng (Flash/ROM)
    file_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"1. Dung lượng file trên đĩa cứng (Flash/ROM): {file_size_mb:.2f} MB")

    # Bắt đầu theo dõi RAM
    tracemalloc.start()

    # Nạp mô hình vào bộ nhớ và cấp phát tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Lấy thông tin kích thước đầu vào
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    # Tạo một bức ảnh rác giả (Dummy input) để chạy thử
    dtype = input_details[0]['dtype']
    if dtype == np.float32:
        dummy_input = np.random.rand(*input_shape).astype(np.float32)
    else: # Dành cho model INT8
        dummy_input = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)

    interpreter.set_tensor(input_details[0]['index'], dummy_input)

    # Chạy suy luận (Inference) lần đầu thường tốn thời gian khởi tạo
    interpreter.invoke()
    
    # Chạy lại lần 2 để đo thời gian chính xác
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()

    # Lấy đỉnh bộ nhớ RAM đã sử dụng
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    print(f"2. Đỉnh RAM tiêu thụ khi chạy (Peak RAM): {peak_mb:.2f} MB")
    print(f"3. Thời gian suy luận 1 ảnh: {(end_time - start_time)*1000:.2f} ms")
    
    if peak_mb <= 30:
        print("=> KẾT LUẬN: ĐẠT! Mô hình cực kỳ an toàn, nằm gọn trong mức RAM 30MB.")
    elif peak_mb <= 35:
        print("=> KẾT LUẬN: CHẤP NHẬN! Mô hình chạm ngưỡng RAM 35MB. Hơi rủi ro nếu hệ thống tốn thêm RAM.")
    else:
        print("=> KẾT LUẬN: KHÔNG ĐẠT! Mô hình vượt mốc 35MB, nguy cơ bị văng (Out of Memory) rất cao.")

if __name__ == "__main__":
    # Đo file Float32 (Mặc định)
    simulate_ram_usage("mobilenetv3_waste.tflite")
    
    # Đo file INT8 Quantized
    simulate_ram_usage("mobilenetv3_waste_INT8.tflite")
