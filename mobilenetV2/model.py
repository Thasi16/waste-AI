import tensorflow as tf
import config
from data_loader import get_data_augmentation

def build_mobilenetv2_model(learning_rate=0.001):
    """
    Khởi tạo mô hình MobileNetV2, kèm theo lớp Data Augmentation ở đầu
    và các lớp phân loại rác ở cuối.
    Trả về mô hình đã được biên dịch (compiled).
    """
    
    # 1. Tải kiến trúc MobileNetV2 pre-train
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=config.IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    
    # 2. Đóng băng toàn bộ base_model ban đầu (Phase 1)
    base_model.trainable = False
    
    # 3. Lấy module tăng cường dữ liệu
    data_augmentation = get_data_augmentation()
    
    # 4. Lắp ráp mô hình hoàn chỉnh
    model = tf.keras.Sequential([
        data_augmentation,
        # Rescaling pixel từ [0, 255] về [-1, 1] cho đúng chuẩn MobileNetV2
        tf.keras.layers.Rescaling(1./127.5, offset=-1),
        
        base_model,
        
        # Phần "đầu não" phân loại
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(config.NUM_CLASSES, activation='softmax')
    ])
    
    # 5. Biên dịch mô hình
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    return model, base_model

def unfreeze_and_recompile(model, base_model, learning_rate=1e-5):
    """
    Thực hiện unfreeze 30 layer cuối cùng của base_model 
    và recompile lại model để chuẩn bị cho bước Fine-Tuning.
    """
    # Mở băng toàn bộ
    base_model.trainable = True
    
    # Tính toán điểm cắt: Đóng băng mọi thứ ngoại trừ 30 layer cuối
    fine_tune_at = len(base_model.layers) - 30
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    # Recompile lại với learning_rate rất nhỏ
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    return model
