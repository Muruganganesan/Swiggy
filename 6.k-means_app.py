import streamlit as st
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



# Load cluster data, model and scaler
clustered_df = pd.read_csv(r"C:\Users\admin\Music\Guvi\Swiggy\clustered_data.csv")

with open(r"C:\Users\admin\Music\Guvi\Swiggy\kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)
with open(r"C:\Users\admin\Music\Guvi\Swiggy\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(r"C:\Users\admin\Music\Guvi\Swiggy\encoder.pkl", "rb") as f:
    encoder = pickle.load(f)   
with open(r"C:\Users\admin\Music\Guvi\Swiggy\scaler_features.pkl", "rb") as f:
    feature_names = pickle.load(f)

st.set_page_config(page_title="📊 KMeans Restaurant Recommender", layout="wide")

st.title("🍽️ Swiggy Restaurant Recommendation System")

# Sidebar UI
st.sidebar.header("🔍 Enter your preferences")
city = st.sidebar.selectbox("City", clustered_df['city'].unique())
cuisine = st.sidebar.selectbox("Cuisine", clustered_df['cuisine'].unique())
rating = st.sidebar.slider("Rating", 1.0, 5.0, 4.0, 0.1)
rating_count = st.sidebar.number_input("Min Rating Count", min_value=0, value=1000)
cost = st.sidebar.number_input("Max Cost", min_value=0, value=300)

if st.sidebar.button("Get Recommendations"):
    with st.spinner("Analyzing preferences..."):
        # 1. Prepare input
        user_input = {
            "city": city,
            "cuisine": cuisine,
            "rating": rating,
            "rating_count": rating_count,
            "cost": cost
        }
        user_df = pd.DataFrame([user_input])

        # 2. Encode and merge
        user_encoded = encoder.transform(user_df[['city', 'cuisine']])
        enc_df = pd.DataFrame(user_encoded, columns=encoder.get_feature_names_out(['city', 'cuisine']))
        num_df = user_df[['rating', 'rating_count', 'cost']].reset_index(drop=True)

        user_vector = pd.concat([num_df, enc_df], axis=1)

        # 3. Reorder columns to match scaler's training
        user_vector = user_vector.reindex(columns=feature_names, fill_value=0)

        # 4. Scale and Predict
        user_scaled = scaler.transform(user_vector)
        cluster = kmeans.predict(user_scaled)[0]

        # 5. Filter by cluster, city, cuisine
        results = clustered_df[
            (clustered_df['cluster'] == cluster) &
            (clustered_df['city'].str.lower() == city.lower()) &
            (clustered_df['cuisine'].str.lower().str.contains(cuisine.lower()))
        ]

        if results.empty:
            st.warning("No exact matches. Showing top from cluster.")
            results = clustered_df[clustered_df['cluster'] == cluster].head(5)
        else:
            results = results.head(5)

        st.subheader("📊 Recommended Restaurants (KMeans)")
        st.dataframe(results[['name', 'city', 'cuisine', 'rating', 'rating_count', 'cost']])
