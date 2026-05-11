import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

import config
import data_loader
import model as model_utils

def plot_history(history_list):
    """Gộp lịch sử Phase 1 và Phase 2 và vẽ biểu đồ"""
    acc = history_list[0].history['accuracy'] + history_list[1].history['accuracy']
    val_acc = history_list[0].history['val_accuracy'] + history_list[1].history['val_accuracy']
    loss = history_list[0].history['loss'] + history_list[1].history['loss']
    val_loss = history_list[0].history['val_loss'] + history_list[1].history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    os.makedirs('image', exist_ok=True)
    plt.savefig('image/training_history.png')
    plt.show()
    plt.close()

def main():
    print("1. Đang nạp toàn bộ danh sách dữ liệu...")
    filepaths, labels = data_loader.load_all_filepaths_and_labels()
    
    print("2. Tách Test Set (Tập kiểm thử vĩnh viễn)...")
    (X_train_val, y_train_val), (X_test, y_test) = data_loader.get_test_set(filepaths, labels)
    
    print("3. Tách Train Set và Validation Set...")
    (X_train, y_train), (X_val, y_val) = data_loader.get_train_val_splits(X_train_val, y_train_val)
    
    print(f"Tổng số ảnh: {len(filepaths)}")
    print(f"Số ảnh để Train: {len(X_train)}")
    print(f"Số ảnh để Validation: {len(X_val)}")
    print(f"Số ảnh cất đi làm Test Set: {len(X_test)}")
    
    # Để sau này test có thể dùng lại X_test, ta lưu danh sách X_test ra file numpy
    np.save('test_filepaths.npy', X_test)
    np.save('test_labels.npy', y_test)

    # --- HUẤN LUYỆN ---
    print("\n4. Chuẩn bị dữ liệu...")
    train_ds = data_loader.create_dataset_from_paths(X_train, y_train, is_training=True)
    val_ds = data_loader.create_dataset_from_paths(X_val, y_val, is_training=False)
    
    model, base_model = model_utils.build_mobilenetv3_model()
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=config.MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ]
    
    # Phase 1
    print("\n--- Phase 1: Trích xuất đặc trưng (Feature Extraction) ---")
    history_p1 = model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS, callbacks=callbacks)
    
    # Phase 2
    print("\n--- Phase 2: Tinh chỉnh sâu (Fine-Tuning) ---")
    model = model_utils.unfreeze_and_recompile(model, base_model)
    history_p2 = model.fit(train_ds, validation_data=val_ds, epochs=config.FINE_TUNE_EPOCHS, callbacks=callbacks)
    
    # Đánh giá cuối cùng
    print(f"\nĐánh giá cuối cùng trên tập Validation:")
    loss, val_acc = model.evaluate(val_ds)
    print(f"-> Độ chính xác của mô hình: {val_acc*100:.2f}%")
    
    # Vẽ biểu đồ
    plot_history([history_p1, history_p2])
    
    print("\n5. Xuất mô hình sang TFLite (Mặc định Float32)...")
    best_model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    tflite_model = converter.convert()
    with open(config.TFLITE_SAVE_PATH, 'wb') as f:
        f.write(tflite_model)
        
    print("\n6. Thực hiện Lượng tử hóa (INT8 Quantization) siêu nhẹ...")
    # Khởi tạo lại converter cho Quantization
    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Cung cấp bộ dữ liệu mẫu (để TFLite tìm được giá trị Min/Max của các biến)
    def representative_dataset():
        # Lấy 100 ảnh mẫu từ tập validation
        for images, _ in val_ds.take(100):
            yield [images]

    converter_int8.representative_dataset = representative_dataset
    
    # Ép toàn bộ các phép tính toán (Ops) về định dạng INT8
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.uint8  # Input là UINT8
    converter_int8.inference_output_type = tf.uint8 # Output là UINT8

    tflite_quant_model = converter_int8.convert()
    
    int8_save_path = config.TFLITE_SAVE_PATH.replace('.tflite', '_INT8.tflite')
    with open(int8_save_path, 'wb') as f:
        f.write(tflite_quant_model)
            
    print(f"HOÀN THÀNH! Đã lưu tại:")
    print(f"- Mô hình Keras: {config.MODEL_SAVE_PATH}")
    print(f"- TFLite (Float32): {config.TFLITE_SAVE_PATH}")
    print(f"- TFLite (INT8): {int8_save_path}")

if __name__ == "__main__":
    main()
