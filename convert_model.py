import tensorflow as tf

# Load your trained Keras model
model = tf.keras.models.load_model('leaf_model.keras')

# Create a TFLite converter object
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimization (this shrinks the model size)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the optimized model to a new file
with open('leaf_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Successfully converted model to leaf_model.tflite")