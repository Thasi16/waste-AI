import os

# --- PATHS ---
DATA_DIR = '../train' # Đã sửa lại đường dẫn vì code giờ nằm trong thư mục con
MODEL_SAVE_PATH = 'mobilenetv1_waste_best.keras'
TFLITE_SAVE_PATH = 'mobilenetv1_waste.tflite'

# --- HYPERPARAMETERS ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15           # Số epoch cho Phase 1 (Feature Extraction)
FINE_TUNE_EPOCHS = 20 # Số epoch cho Phase 2 (Fine-Tuning)
NUM_CLASSES = 5

# Khớp hoàn toàn với tên thư mục con trong 'train'
CLASS_NAMES = ['bag_other', 'metal', 'mixed', 'paper', 'plastic']

# Data Split Settings
TEST_SPLIT = 0.1      # 10% dữ liệu sẽ bị cắt ra vĩnh viễn làm Test Set
VAL_SPLIT = 0.15      # 15% của 90% còn lại sẽ dùng làm Validation Set
