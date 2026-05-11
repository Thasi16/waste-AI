import os

# Lấy đường dẫn gốc của thư mục cha (thư mục chứa dataset)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- PATHS ---
DATA_DIR = os.path.join(ROOT_DIR, 'train')
MODEL_SAVE_PATH = 'mobilenetv3_waste_best.keras'
TFLITE_SAVE_PATH = 'mobilenetv3_waste.tflite'

# --- HYPERPARAMETERS ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15           # Số epoch cho Phase 1 (Feature Extraction)
FINE_TUNE_EPOCHS = 20 # Số epoch cho Phase 2 (Fine-Tuning)
NUM_CLASSES = 5

# Khớp hoàn toàn với tên thư mục con trong 'train'
CLASS_NAMES = ['bag_other', 'metal', 'mixed', 'paper', 'plastic']

# Data Split Settings
TEST_SPLIT = 0.1      
VAL_SPLIT = 0.15      
RANDOM_SEED = 16 
