import tensorflow as tf
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

import config

def get_data_augmentation():
    """Trả về mô hình Data Augmentation để tăng cường dữ liệu"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

def load_all_filepaths_and_labels():
    """Đọc toàn bộ đường dẫn ảnh và gán nhãn từ thư mục DATA_DIR"""
    filepaths = []
    labels = []
    
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = os.path.join(config.DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                filepaths.append(os.path.join(class_dir, img_name))
                labels.append(class_idx)
                
    return np.array(filepaths), np.array(labels)

def get_test_set(filepaths, labels):
    """
    Tách ra một tập Test Set (không bao giờ dùng để train hay validation)
    Trả về: (train_val_paths, train_val_labels), (test_paths, test_labels)
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        filepaths, labels, 
        test_size=config.TEST_SPLIT, 
        stratify=labels, # Đảm bảo chia đều các loại rác
        random_state=config.RANDOM_SEED
    )
    return (X_train_val, y_train_val), (X_test, y_test)

def create_dataset_from_paths(filepaths, labels, is_training=False):
    """Tạo đối tượng tf.data.Dataset từ danh sách đường dẫn và nhãn"""
    
    def parse_function(filename, label):
        # Đọc file ảnh từ ổ cứng
        image_string = tf.io.read_file(filename)
        # Giải mã ảnh
        image = tf.image.decode_jpeg(image_string, channels=3)
        # Resize về IMG_SIZE
        image = tf.image.resize(image, config.IMG_SIZE)
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Cache vào RAM/ổ đĩa để chạy nhanh hơn ở các epoch sau
    dataset = dataset.cache()
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000, seed=config.RANDOM_SEED)
        
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

def get_train_val_splits(X_train_val, y_train_val):
    """
    Chia tập X_train_val còn lại thành tập Train và tập Validation.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=config.VAL_SPLIT, 
        stratify=y_train_val,
        random_state=config.RANDOM_SEED
    )
    return (X_train, y_train), (X_val, y_val)
