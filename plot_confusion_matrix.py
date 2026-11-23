
import matplotlib.pyplot as plt
import numpy as np

# Actual confusion matrix values
cm = np.array([
    [401, 99],
    [112, 388]
])

# Class names
class_names = ["FAKE", "REAL"]

# Create confusion matrix graph
plt.figure(figsize=(7, 6))

plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("VGG19 - Confusion Matrix", fontsize=16, pad=15)

plt.colorbar()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("Actual Label", fontsize=12)

# Display values inside each cell
threshold = cm.max() / 2

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, str(cm[i, j]),
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
            fontsize=16
        )

plt.tight_layout()

# Save graph
plt.savefig(
    "vgg19_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Confusion matrix saved successfully!")
