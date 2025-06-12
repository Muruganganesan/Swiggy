import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# === Load cleaned data ===
cleaned_df = pd.read_csv(r"C:\Users\admin\Music\Guvi\Swiggy\cleaned_data.csv")
encoded_df = pd.read_csv(r"C:\Users\admin\Music\Guvi\Swiggy\encoded_data.csv")

# Load encoder
with open(r"C:\Users\admin\Music\Guvi\Swiggy\encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# === Streamlit App UI ===
st.title("🍽️ Swiggy Restaurant Recommendation System")

st.sidebar.header("🔍 Enter your preferences")

city = st.sidebar.selectbox("City", cleaned_df['city'].unique())
cuisine = st.sidebar.selectbox("Cuisine", cleaned_df['cuisine'].unique())
rating = st.sidebar.slider("Minimum Rating", 1.0, 5.0, 4.0, 0.1)
rating_count = st.sidebar.number_input("Minimum Rating Count", min_value=0, value=1000)
cost = st.sidebar.number_input("Maximum Cost", min_value=0, value=300)

if st.sidebar.button("Get Recommendations"):
    # 1. Prepare user input
    user_input = {
        'city': city,
        'cuisine': cuisine,
        'rating': rating,
        'rating_count': rating_count,
        'cost': cost
    }
    user_df = pd.DataFrame([user_input])

    # 2. Step 1: Partial cuisine match (case insensitive substring)
    mask = (cleaned_df['city'].str.lower() == city.lower()) & \
           (cleaned_df['cuisine'].str.lower().str.contains(cuisine.lower()))
    filtered_cleaned_df = cleaned_df[mask]
    filtered_encoded_df = encoded_df[mask]

    # 3. Step 2: If no match, fallback to city-based filtering only
    if filtered_encoded_df.empty:
        st.info("No exact cuisine match found. Showing top restaurants from selected city only.")
        city_mask = (cleaned_df['city'].str.lower() == city.lower())
        filtered_cleaned_df = cleaned_df[city_mask]
        filtered_encoded_df = encoded_df[city_mask]

    # 4. Again, if still empty, no recommendation possible
    if filtered_encoded_df.empty:
        st.error("😕 Sorry, no restaurants found for your selected filters.")
    else:
        # 5. Encode user input
        user_encoded = encoder.transform(user_df[['city', 'cuisine']])
        user_numeric = user_df[['rating', 'rating_count', 'cost']].values
        user_final = pd.concat([
            pd.DataFrame(user_numeric),
            pd.DataFrame(user_encoded)
        ], axis=1)

        # 6. Compute cosine similarity
        encoded_df_numeric = filtered_encoded_df.select_dtypes(include=['number'])
        user_final_numeric = user_final.select_dtypes(include=['number'])

        similarities = cosine_similarity(user_final_numeric, encoded_df_numeric)

        # 7. Top N results
        top_indices = similarities[0].argsort()[::-1][:5]
        recommendations = filtered_cleaned_df.iloc[top_indices]

        # 8. Display recommendations
        st.subheader("🎯 Top Recommended Restaurants")
        st.table(recommendations[['name', 'city', 'cuisine', 'rating', 'rating_count', 'cost']])
