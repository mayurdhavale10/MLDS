import pickle

model_path = "artifacts/model.pkl"  # Ensure this path is correct

with open(model_path, "rb") as file:
    model = pickle.load(file)

print("✅ Model Loaded Successfully!")
print("🔍 Expected Input Feature Count:", model.n_features_in_)  # Should match transformed data shape
