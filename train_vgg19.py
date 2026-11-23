
import tensorflow as tf
import os

from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras import layers, models

# Dataset path
DATASET_DIR = "/mnt/c/Users/Hp/Downloads/Real vs Fake Faces/Real vs Fake Faces/8020"
train_dir = os.path.join(DATASET_DIR, "train")
valid_dir = os.path.join(DATASET_DIR, "valid")
test_dir = os.path.join(DATASET_DIR, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 8

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["fake", "real"]
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["fake", "real"],
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["fake", "real"],
    shuffle=False
)

# Preprocess images for VGG19
train_ds = train_ds.map(
    lambda x, y: (preprocess_input(x), y)
)
valid_ds = valid_ds.map(
    lambda x, y: (preprocess_input(x), y)
)
test_ds = test_ds.map(
    lambda x, y: (preprocess_input(x), y)
)

# Load pre-trained VGG19
base_model = VGG19(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze VGG19 layers
base_model.trainable = False

# Build classification model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("VGG19 model created successfully!")
# Train the model
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=3
)

# Save the trained model
model.save("vgg19_deepfake_model.keras")

print("Model training completed and saved successfully!")
