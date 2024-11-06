from tensorflow.keras.models import load_model

# Specify the path to your saved model
model_path = "model/best_model.h5"  # For .h5 model files
# or
# model_path = "path/to/your/saved_model_directory"  # For SavedModel format

# Load the model
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# (Optional) Check model summary to verify successful load
model.summary()