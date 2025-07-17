import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from data_setup import get_data_generators
from config import MODEL_PATH

# 🔄 Load validation generator only
_, val_generator = get_data_generators()
val_generator.reset()  # Ensure predictions line up with labels

# 💾 Load saved model
model = load_model(MODEL_PATH)

# 🧠 Run predictions
predictions = model.predict(val_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_generator.classes
labels = list(val_generator.class_indices.keys())

# ✅ Ensure label count matches class indices
label_indices = list(range(len(labels)))

# 📊 Print performance metrics
print("✅ Unique true classes:", sorted(set(true_classes)))
print("✅ Unique predicted classes:", sorted(set(predicted_classes)))
print("✅ Total labels:", len(labels))

print(classification_report(
    true_classes,
    predicted_classes,
    labels=label_indices,
    target_names=labels
))
