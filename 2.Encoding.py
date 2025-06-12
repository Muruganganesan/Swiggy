from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import pickle

# Load cleaned data
df = pd.read_csv(r"C:\Users\admin\Music\Guvi\Swiggy\cleaned_data.csv")

# Select categorical columns
categorical_cols = ['city', 'cuisine']

# Create encoder with correct argument
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit and transform
encoded_array = encoder.fit_transform(df[categorical_cols])

# Convert to DataFrame
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))

# Drop original categorical columns
df_numeric = df.drop(columns=categorical_cols)

# Combine numerical and encoded
final_df = pd.concat([df_numeric.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Save to CSV
final_df.to_csv("encoded_data.csv", index=False)
print("✅ Encoded data saved as 'encoded_data.csv'")

# Save encoder as pickle
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
print("✅ Encoder saved as 'encoder.pkl'")
