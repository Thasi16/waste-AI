import json
import os

with open(r"c:\Users\Admin\Downloads\smart waste dataset.v1i.folder\mobilenetV2\generate_kaggle_notebook.py", "r", encoding='utf-8') as f:
    nb = json.load(f)

# --- UPDATE FOR V3 ---
nb3 = json.loads(json.dumps(nb))
nb3['cells'][0]['source'][0] = "# 🚀 Huấn luyện MobileNetV3 (Waste Classification) trên Kaggle\n"
nb3['cells'][2]['source'][4] = "MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'mobilenetv3_waste_best.keras')\n"
nb3['cells'][2]['source'][5] = "QAT_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'mobilenetv3_waste_qat.h5')\n"
nb3['cells'][2]['source'][6] = "TFLITE_SAVE_PATH = os.path.join(OUTPUT_DIR, 'mobilenetv3_waste.tflite')\n"

nb3['cells'][4]['source'] = [
    "# --- 3. MODEL ARCHITECTURE ---\n",
    "\n",
    "def build_mobilenetv3_model(learning_rate=0.001):\n",
    "    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))\n",
    "    x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(inputs)\n",
    "    \n",
    "    base_model = tf.keras.applications.MobileNetV3Large(\n",
    "        input_tensor=x,\n",
    "        include_preprocessing=False,\n",
    "        include_top=False,\n",
    "        weights='imagenet'\n",
    "    )\n",
    "    \n",
    "    for layer in base_model.layers:\n",
    "        layer.trainable = False\n",
    "        \n",
    "    x = base_model.output\n",
    "    x = tf.keras.layers.GlobalAveragePooling2D()(x)\n",
    "    x = tf.keras.layers.Dropout(0.2)(x)\n",
    "    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)\n",
    "    \n",
    "    model = tf.keras.Model(inputs=inputs, outputs=outputs)\n",
    "    \n",
    "    model.compile(\n",
    "        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),\n",
    "        loss=tf.keras.losses.SparseCategoricalCrossentropy(),\n",
    "        metrics=['accuracy']\n",
    "    )\n",
    "    return model, base_model\n",
    "\n",
    "def unfreeze_and_recompile(model, base_model, learning_rate=1e-5):\n",
    "    for layer in base_model.layers:\n",
    "        layer.trainable = True\n",
    "        \n",
    "    fine_tune_at = len(base_model.layers) - 30\n",
    "    for layer in base_model.layers[:fine_tune_at]:\n",
    "        layer.trainable = False\n",
    "        \n",
    "    model.compile(\n",
    "        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),\n",
    "        loss=tf.keras.losses.SparseCategoricalCrossentropy(),\n",
    "        metrics=['accuracy']\n",
    "    )\n",
    "    return model"
]

for i, line in enumerate(nb3['cells'][5]['source']):
    if 'build_mobilenetv2_model' in line:
        nb3['cells'][5]['source'][i] = line.replace('build_mobilenetv2_model', 'build_mobilenetv3_model')

v3_dir = r"c:\Users\Admin\Downloads\smart waste dataset.v1i.folder\mobilenetV3"
with open(os.path.join(v3_dir, 'kaggle_train_mobilenetv3_QAT.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(nb3, f, indent=2, ensure_ascii=False)

v3_py = f"import json\nimport os\n\nnotebook = {json.dumps(nb3, indent=2, ensure_ascii=False)}\n\nwith open(os.path.join(os.path.dirname(__file__), 'kaggle_train_mobilenetv3_QAT.ipynb'), 'w', encoding='utf-8') as f:\n    json.dump(notebook, f, indent=2, ensure_ascii=False)\nprint('Successfully generated kaggle_train_mobilenetv3_QAT.ipynb')"
with open(os.path.join(v3_dir, 'generate_kaggle_notebook.py'), 'w', encoding='utf-8') as f:
    f.write(v3_py)


# --- UPDATE FOR V1 ---
nb1 = json.loads(json.dumps(nb))
nb1['cells'][0]['source'][0] = "# 🚀 Huấn luyện MobileNetV1 (Waste Classification) trên Kaggle\n"
nb1['cells'][2]['source'][4] = "MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'mobilenetv1_waste_best.keras')\n"
nb1['cells'][2]['source'][5] = "QAT_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'mobilenetv1_waste_qat.h5')\n"
nb1['cells'][2]['source'][6] = "TFLITE_SAVE_PATH = os.path.join(OUTPUT_DIR, 'mobilenetv1_waste.tflite')\n"

nb1['cells'][4]['source'] = [
    "# --- 3. MODEL ARCHITECTURE ---\n",
    "\n",
    "def build_mobilenetv1_model(learning_rate=0.001):\n",
    "    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))\n",
    "    x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(inputs)\n",
    "    \n",
    "    base_model = tf.keras.applications.MobileNet(\n",
    "        input_tensor=x,\n",
    "        include_top=False,\n",
    "        weights='imagenet'\n",
    "    )\n",
    "    \n",
    "    for layer in base_model.layers:\n",
    "        layer.trainable = False\n",
    "        \n",
    "    x = base_model.output\n",
    "    x = tf.keras.layers.GlobalAveragePooling2D()(x)\n",
    "    x = tf.keras.layers.Dropout(0.2)(x)\n",
    "    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)\n",
    "    \n",
    "    model = tf.keras.Model(inputs=inputs, outputs=outputs)\n",
    "    \n",
    "    model.compile(\n",
    "        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),\n",
    "        loss=tf.keras.losses.SparseCategoricalCrossentropy(),\n",
    "        metrics=['accuracy']\n",
    "    )\n",
    "    return model, base_model\n",
    "\n",
    "def unfreeze_and_recompile(model, base_model, learning_rate=1e-5):\n",
    "    for layer in base_model.layers:\n",
    "        layer.trainable = True\n",
    "        \n",
    "    fine_tune_at = len(base_model.layers) - 30\n",
    "    for layer in base_model.layers[:fine_tune_at]:\n",
    "        layer.trainable = False\n",
    "        \n",
    "    model.compile(\n",
    "        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),\n",
    "        loss=tf.keras.losses.SparseCategoricalCrossentropy(),\n",
    "        metrics=['accuracy']\n",
    "    )\n",
    "    return model"
]

for i, line in enumerate(nb1['cells'][5]['source']):
    if 'build_mobilenetv2_model' in line:
        nb1['cells'][5]['source'][i] = line.replace('build_mobilenetv2_model', 'build_mobilenetv1_model')

v1_dir = r"c:\Users\Admin\Downloads\smart waste dataset.v1i.folder\mobilenetV1"
os.makedirs(v1_dir, exist_ok=True)
with open(os.path.join(v1_dir, 'kaggle_train_mobilenetv1_QAT.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(nb1, f, indent=2, ensure_ascii=False)

v1_py = f"import json\nimport os\n\nnotebook = {json.dumps(nb1, indent=2, ensure_ascii=False)}\n\nwith open(os.path.join(os.path.dirname(__file__), 'kaggle_train_mobilenetv1_QAT.ipynb'), 'w', encoding='utf-8') as f:\n    json.dump(notebook, f, indent=2, ensure_ascii=False)\nprint('Successfully generated kaggle_train_mobilenetv1_QAT.ipynb')"
with open(os.path.join(v1_dir, 'generate_kaggle_notebook.py'), 'w', encoding='utf-8') as f:
    f.write(v1_py)
