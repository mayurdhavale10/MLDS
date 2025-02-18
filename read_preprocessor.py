import pickle

preprocessor_path = "artifacts/preprocessor.pkl"

with open(preprocessor_path, "rb") as f:
    preprocessor = pickle.load(f)

print("✅ Preprocessor Loaded Successfully!")
print("🔎 Expected Features:", preprocessor.feature_names_in_)  # Check the expected feature names
