
import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.applications.vgg19 import preprocess_input

# Dataset and model paths
DATASET_DIR = "/mnt/c/Users/Hp/Downloads/Real vs Fake Faces/Real vs Fake Faces/8020"

TEST_DIR = os.path.join(DATASET_DIR, "test")
MODEL_PATH = "vgg19_deepfake_model.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 8

# Step 1: Load the trained model
print("Loading trained VGG19 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# Step 2: Load the test dataset
print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["fake", "real"],
    shuffle=False
)

# Step 3: Apply VGG19 preprocessing
test_ds = test_ds.map(
    lambda x, y: (preprocess_input(x), y)
)

# Step 4: Evaluate the model
print("\nEvaluating VGG19 model...")

test_loss, test_accuracy = model.evaluate(test_ds)

print("\nTest Results")
print("-------------------------")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Step 5: Get actual labels
y_true = np.concatenate([
    labels.numpy().reshape(-1)
    for images, labels in test_ds
]).astype(int)

# Step 6: Predict test images
print("\nPredicting test images...")

y_prob = model.predict(test_ds, verbose=1).reshape(-1)

# Sigmoid threshold: 0.5
y_pred = (y_prob >= 0.5).astype(int)

# Step 7: Calculate confusion matrix
cm = np.zeros((2, 2), dtype=int)

for actual, predicted in zip(y_true, y_pred):
    cm[actual, predicted] += 1

tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix")
print("-------------------------")
print("                 Predicted")
print("              FAKE     REAL")
print(f"Actual FAKE    {tn:5d}    {fp:5d}")
print(f"Actual REAL    {fn:5d}    {tp:5d}")

# Step 8: Calculate performance metrics
accuracy = (tp + tn) / max(len(y_true), 1)

fake_precision = tn / max(tn + fn, 1)
fake_recall = tn / max(tn + fp, 1)
fake_f1 = (
    2 * fake_precision * fake_recall
    / max(fake_precision + fake_recall, 1e-10)
)

real_precision = tp / max(tp + fp, 1)
real_recall = tp / max(tp + fn, 1)
real_f1 = (
    2 * real_precision * real_recall
    / max(real_precision + real_recall, 1e-10)
)

macro_precision = (fake_precision + real_precision) / 2
macro_recall = (fake_recall + real_recall) / 2
macro_f1 = (fake_f1 + real_f1) / 2

print("\nClassification Report")
print("-------------------------")
print(f"{'Class':<10}{'Precision':<12}{'Recall':<12}{'F1-score'}")

print(
    f"{'FAKE':<10}"
    f"{fake_precision:.4f}      "
    f"{fake_recall:.4f}      "
    f"{fake_f1:.4f}"
)

print(
    f"{'REAL':<10}"
    f"{real_precision:.4f}      "
    f"{real_recall:.4f}      "
    f"{real_f1:.4f}"
)

print("\nOverall Performance")
print("-------------------------")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {macro_precision:.4f}")
print(f"Recall:    {macro_recall:.4f}")
print(f"F1-score:  {macro_f1:.4f}")

print("\nVGG19 model evaluation completed successfully!")
