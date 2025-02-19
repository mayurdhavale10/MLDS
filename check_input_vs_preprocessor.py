import pickle
import pandas as pd

# ✅ Load the preprocessor.pkl file
preprocessor_path = "artifacts/preprocessor.pkl"

try:
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
    
    # Extract OneHotEncoder from the preprocessor pipeline
    ohe = preprocessor.named_transformers_['cat_pipeline'].named_steps['one_hot_encoder']
    
    print("\n✅ Valid Categories in OneHotEncoder:")
    for i, categories in enumerate(ohe.categories_):
        print(f"Column {i}: {categories}")

    # ✅ Create a test input based on expected column names
    sample_input = pd.DataFrame({
        'gender': ['male'],  
        'race/ethnicity': ['group B'],  
        'parental level of education': ["bachelor's degree"],  
        'lunch': ['standard'],  
        'test preparation course': ['none'],  
        'reading score': [80],  
        'writing score': [85]  
    })

    print("\n🔍 Checking if input data matches expected columns...")
    expected_columns = preprocessor.feature_names_in_

    # Check for missing or extra columns
    missing_cols = set(expected_columns) - set(sample_input.columns)
    extra_cols = set(sample_input.columns) - set(expected_columns)

    if missing_cols:
        print(f"❌ Missing columns in input data: {missing_cols}")
    if extra_cols:
        print(f"⚠️ Extra columns in input data (shouldn't be there): {extra_cols}")

    # ✅ Try transforming the sample input
    try:
        transformed_data = preprocessor.transform(sample_input)
        print("\n✅ Input data successfully transformed!")
    except Exception as e:
        print(f"\n❌ Error transforming input data: {e}")

except Exception as e:
    print(f"\n❌ Error loading preprocessor.pkl: {e}")
