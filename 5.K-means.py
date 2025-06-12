import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
import pickle

# Load cleaned and encoded data
df = pd.read_csv(r"C:\Users\admin\Music\Guvi\Swiggy\cleaned_data.csv")
encoded_df = pd.read_csv(r"C:\Users\admin\Music\Guvi\Swiggy\encoded_data.csv")

# Use only numerical features
X = encoded_df.select_dtypes(include='number')
feature_names = X.columns.tolist()  # Save feature names

# Fit scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster to cleaned_df
df['cluster'] = cluster_labels
df.to_csv("clustered_data.csv", index=False)

# Save all models
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

with open("scaler_features.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("✅ Training completed. Models and clustered data saved.")
