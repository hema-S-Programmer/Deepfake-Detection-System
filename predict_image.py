
import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.applications.vgg19 import preprocess_input

# Load the trained model
model = tf.keras.models.load_model("vgg19_deepfake_model.keras")

# Get image path from the terminal
image_path = sys.argv[1]

# Load and resize the image
img = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

# Convert image into an array
img_array = tf.keras.utils.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Apply VGG19 preprocessing
img_array = preprocess_input(img_array)

# Predict the image
prediction = model.predict(img_array, verbose=0)[0][0]

# Display the result
if prediction >= 0.5:
    print("Prediction: REAL")
    print(f"Model score: {prediction:.4f}")
else:
    print("Prediction: FAKE")
    print(f"Model score: {prediction:.4f}")
