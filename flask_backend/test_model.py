import tensorflow as tf
try:
    model = tf.keras.models.load_model("water_cnn_model.h5")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")