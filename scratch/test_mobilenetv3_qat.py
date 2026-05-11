import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np

# Create a minimal MobileNetV3
model = tf.keras.applications.MobileNetV3Small(input_shape=(224, 224, 3), weights=None, classes=10)

def apply_quantization_to_layer(layer):
    # Only annotate Conv2D, DepthwiseConv2D, Dense
    if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Dense)):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer

print("Cloning annotated model...")
annotated_model = tf.keras.models.clone_model(
    model,
    clone_function=apply_quantization_to_layer
)

print("Applying quantization...")
try:
    q_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
