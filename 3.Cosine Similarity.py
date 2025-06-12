import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load data
cleaned_df = pd.read_csv(r"C:\Users\admin\Music\Guvi\Swiggy\cleaned_data.csv")
encoded_df = pd.read_csv(r"C:\Users\admin\Music\Guvi\Swiggy\encoded_data.csv")

# Load encoder
with open(r"C:\Users\admin\Music\Guvi\Swiggy\encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# === USER INPUT ===
user_input = {
    'city': 'Chennai',
    'cuisine': 'South Indian',
    'rating': 4.0,
    'rating_count': 1200,
    'cost': 300
}

# 1. Convert input to DataFrame
user_df = pd.DataFrame([user_input])

# 2. Encode categorical columns
user_encoded = encoder.transform(user_df[['city', 'cuisine']])

# 3. Combine with numeric values
user_numeric = user_df[['rating', 'rating_count', 'cost']].values
user_final = pd.DataFrame(
    data = pd.concat([
        pd.DataFrame(user_numeric),
        pd.DataFrame(user_encoded)
    ], axis=1)
)

# 4. Compute cosine similarity

# ✅ FIX: Select only numeric columns (avoid strings like 'name', 'address')
encoded_df_numeric = encoded_df.select_dtypes(include=['number'])
user_final_numeric = user_final.select_dtypes(include=['number'])

# ✅ Now safe to compute similarity
similarities = cosine_similarity(user_final_numeric, encoded_df_numeric)

# 5. Get top 5 matching restaurant indices
top_indices = similarities[0].argsort()[::-1][:5]

# 6. Show recommendations from original data
recommended_restaurants = cleaned_df.iloc[top_indices]

# Display
print("🍽️ Top Recommended Restaurants:")
print(recommended_restaurants[['name', 'city', 'cuisine', 'rating', 'rating_count', 'cost']])

