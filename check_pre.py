import pickle

preprocessor_path = "artifacts/preprocessor.pkl"

with open(preprocessor_path, "rb") as f:
    preprocessor = pickle.load(f)

print("✅ Expected Features in Preprocessor:")
print(preprocessor.feature_names_in_)  # This should match your dataset columns
