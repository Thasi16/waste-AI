import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import config
import data_loader

print("Đang tải mô hình tốt nhất từ K-Fold...")
model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)

def evaluate_full_test_set():
    """Chạy đánh giá trên toàn bộ tập Test (20% cất đi) để ra Báo Cáo Phân Loại"""
    if not os.path.exists('test_filepaths.npy') or not os.path.exists('test_labels.npy'):
        print("Lỗi: Không tìm thấy file lưu Test Set. Vui lòng chạy train.py trước!")
        return
        
    X_test = np.load('test_filepaths.npy')
    y_test = np.load('test_labels.npy')
    
    print(f"\nĐang đánh giá trên {len(X_test)} ảnh của tập Test Set...")
    test_ds = data_loader.create_dataset_from_paths(X_test, y_test, is_training=False)
    
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Độ chính xác (Accuracy) trên tập Test: {acc*100:.2f}%\n")
    
    # Dự đoán để in Classification Report
    y_pred = []
    y_true = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
        
    print("=== BÁO CÁO PHÂN LOẠI TRÊN TẬP TEST ===")
    print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES))
    
    print("\nĐang hiển thị Ma trận nhầm lẫn (Confusion Matrix)...")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.CLASS_NAMES)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title('Confusion Matrix (Ma trận nhầm lẫn)')
    os.makedirs('image', exist_ok=True)
    plt.savefig('image/confusion_matrix.png')
    # Xóa plt.show() ở đây để không chặn tiến trình
    print("Đã lưu Ma trận nhầm lẫn tại 'image/confusion_matrix.png'")

def predict_random_test_batch(num_images=6):
    """Lấy ngẫu nhiên vài ảnh từ tập Test để hiển thị trực quan"""
    if not os.path.exists('test_filepaths.npy'):
        print("Lỗi: Không tìm thấy file lưu Test Set.")
        return
        
    X_test = np.load('test_filepaths.npy')
    y_test = np.load('test_labels.npy')
    
    print(f"\nĐang lấy {num_images} ảnh ngẫu nhiên từ tập Test để vẽ hình...")
    # Bốc ngẫu nhiên index
    random_idx = np.random.choice(len(X_test), num_images, replace=False)
    X_random = X_test[random_idx]
    y_random = y_test[random_idx]
    
    test_ds = data_loader.create_dataset_from_paths(X_random, y_random, is_training=False)
    
    # Ở đây do test_ds chứa đúng num_images, lấy mẻ đầu tiên là đủ
    for images, labels in test_ds.take(1):
        predictions = model.predict(images, verbose=0)
        
        plt.figure(figsize=(12, 8))
        for i in range(num_images):
            # Cẩn thận nếu batch_size của test_ds nhỏ hơn num_images, nhưng ta đảm bảo ds này đủ hình
            if i >= len(images):
                break
                
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            
            predicted_class_idx = np.argmax(predictions[i])
            predicted_prob = np.max(predictions[i]) * 100
            predicted_class = config.CLASS_NAMES[predicted_class_idx]
            
            actual_class_idx = labels[i].numpy()
            actual_class = config.CLASS_NAMES[actual_class_idx]
            
            title_color = "green" if predicted_class == actual_class else "red"
            
            plt.title(f"Dự đoán: {predicted_class} ({predicted_prob:.1f}%)\nThực tế: {actual_class}", color=title_color)
            plt.axis("off")
            
        plt.tight_layout()
    os.makedirs('image', exist_ok=True)
    plt.savefig('image/test_predictions.png')
    print("Đã lưu kết quả tại 'image/test_predictions.png'")

def predict_single_image(image_path):
    """Dự đoán 1 tấm ảnh bên ngoài (ảnh tải trên mạng về)"""
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy ảnh tại {image_path}")
        return

    print(f"\nĐang dự đoán ảnh: {image_path}...")
    img = tf.keras.utils.load_img(image_path, target_size=config.IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_prob = np.max(predictions[0]) * 100
    predicted_class = config.CLASS_NAMES[predicted_class_idx]

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Dự đoán: {predicted_class} ({predicted_prob:.1f}%)", color="blue")
    plt.axis("off")
    plt.show()
    print(f"==> Kết quả: {predicted_class} (Độ tự tin: {predicted_prob:.1f}%)")


if __name__ == "__main__":
    # 1. Đánh giá toàn bộ tập Test (Chạy classification report)
    evaluate_full_test_set()
    
    # 2. Rút ngẫu nhiên 6 tấm trong tập Test ra vẽ ảnh
    predict_random_test_batch(num_images=6)
    
    # 3. Hiển thị TOÀN BỘ các cửa sổ hình ảnh cùng một lúc
    plt.show()
    
    # 4. Test ảnh bên ngoài (Bỏ comment để dùng)
    # predict_single_image(r'C:\Users\Admin\Downloads\anh_rac_test.jpg')
