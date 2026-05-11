import tensorflow as tf
import config
import data_loader
import os

print("1. Đang nạp lại Validation Dataset để làm mẫu (Representative Dataset)...")
# Cần lấy mẫu data để TFLite biết dải giá trị màu sắc mà lượng tử hóa
filepaths, labels = data_loader.load_all_filepaths_and_labels()
(X_train_val, y_train_val), (X_test, y_test) = data_loader.get_test_set(filepaths, labels)
(X_train, y_train), (X_val, y_val) = data_loader.get_train_val_splits(X_train_val, y_train_val)
val_ds = data_loader.create_dataset_from_paths(X_val, y_val, is_training=False)

print("2. Đang nạp mô hình Keras tốt nhất...")
best_model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)

print("3. Khởi tạo thuật toán Lượng tử hóa (INT8)...")
converter_int8 = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset():
    # Lấy 100 ảnh mẫu từ tập validation
    for images, _ in val_ds.take(100):
        yield [images]

converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.uint8  # Input là UINT8
converter_int8.inference_output_type = tf.uint8 # Output là UINT8

print("4. Bắt đầu ép kiểu (Quá trình này mất khoảng vài chục giây)...")
tflite_quant_model = converter_int8.convert()

int8_save_path = config.TFLITE_SAVE_PATH.replace('.tflite', '_INT8.tflite')
with open(int8_save_path, 'wb') as f:
    f.write(tflite_quant_model)
        
print(f"HOÀN THÀNH! Đã nén thành công file: {int8_save_path}")
