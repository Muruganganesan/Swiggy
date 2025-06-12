import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ===== Page config =====
st.set_page_config(page_title="🍽️ Smart Restaurant Recommender", layout="wide")

# ===== Load Data and Models =====
@st.cache_data
def load_data():
    cleaned = pd.read_csv("cleaned_data.csv")
    encoded = pd.read_csv("encoded_data.csv")
    clustered = pd.read_csv("clustered_data.csv")
    return cleaned, encoded, clustered

@st.cache_resource
def load_models():
    with open("encoder.pkl", "rb") as f1:
        encoder = pickle.load(f1)
    with open("scaler.pkl", "rb") as f2:
        scaler = pickle.load(f2)
    with open("kmeans_model.pkl", "rb") as f3:
        kmeans = pickle.load(f3)
    with open("scaler_features.pkl", "rb") as f4:
        features = pickle.load(f4)
    return encoder, scaler, kmeans, features

cleaned_df, encoded_df, clustered_df = load_data()
encoder, scaler, kmeans, feature_names = load_models()

# ===== App UI =====
st.title("🍽️ Swiggy Restaurant Recommendation System")

st.sidebar.header("🔍 User Preferences")
method = st.sidebar.radio("Recommendation Method", ["Cosine Similarity", "KMeans Clustering"])
city = st.sidebar.selectbox("City", cleaned_df["city"].unique())
cuisine = st.sidebar.selectbox("Cuisine", cleaned_df["cuisine"].unique())
rating = st.sidebar.slider("Minimum Rating", 1.0, 5.0, 4.0, 0.1)
rating_count = st.sidebar.number_input("Minimum Rating Count", value=1000, step=100)
cost = st.sidebar.number_input("Maximum Cost", value=300, step=50)

if st.sidebar.button("Get Recommendations"):
    with st.spinner("🔍 Finding the best restaurants for you..."):
        # ===== Input preparation =====
        user_input = {
            "city": city,
            "cuisine": cuisine,
            "rating": rating,
            "rating_count": rating_count,
            "cost": cost
        }
        user_df = pd.DataFrame([user_input])

        # Encode
        user_encoded = encoder.transform(user_df[['city', 'cuisine']])
        enc_df = pd.DataFrame(user_encoded, columns=encoder.get_feature_names_out(['city', 'cuisine']))
        num_df = user_df[['rating', 'rating_count', 'cost']]
        user_vector = pd.concat([num_df, enc_df], axis=1)

        recommendations = pd.DataFrame()

        if method == "Cosine Similarity":
            # Filter
            mask = (cleaned_df["city"].str.lower() == city.lower()) & \
                   (cleaned_df["cuisine"].str.lower().str.contains(cuisine.lower()))
            filtered_cleaned = cleaned_df[mask]
            filtered_encoded = encoded_df[mask]

            if filtered_encoded.empty:
                st.warning("No exact cuisine match. Showing all in city.")
                mask = (cleaned_df["city"].str.lower() == city.lower())
                filtered_cleaned = cleaned_df[mask]
                filtered_encoded = encoded_df[mask]

            if filtered_encoded.empty:
                st.error("❌ No restaurants found.")
            else:
                sim = cosine_similarity(user_vector, filtered_encoded.select_dtypes(include='number'))
                top_indices = sim[0].argsort()[::-1][:5]
                recommendations = filtered_cleaned.iloc[top_indices]
                st.subheader("🎯 Recommended Restaurants (Cosine Similarity)")

        elif method == "KMeans Clustering":
            # Reindex & scale
            user_vector = user_vector.reindex(columns=feature_names, fill_value=0)
            user_scaled = scaler.transform(user_vector)
            pred_cluster = kmeans.predict(user_scaled)[0]
            cluster_df = clustered_df[clustered_df["cluster"] == pred_cluster]

            # Clean 'cost' column
            cluster_df["cost"] = (
                cluster_df["cost"]
                .astype(str)
                .str.replace("₹", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            cluster_df["cost"] = pd.to_numeric(cluster_df["cost"], errors="coerce")

            # Smart cuisine match (multi-part)
            user_parts = [x.strip().lower() for x in cuisine.split(",")]
            def match_any(c):
                return any(part in c.lower() for part in user_parts)

            filtered = cluster_df[
                (cluster_df["city"].str.lower() == city.lower()) &
                (cluster_df["rating"] >= rating) &
                (cluster_df["rating_count"] >= rating_count) &
                (cluster_df["cost"] <= cost) &
                (cluster_df["cuisine"].apply(match_any))
            ]

            if filtered.empty:
                st.warning("⚠️ No restaurants matched all filters. Showing top in cluster.")
                recommendations = cluster_df.head(5)
            else:
                recommendations = filtered.head(5)

            st.subheader("📊 Recommended Restaurants (KMeans)")

        # ===== Output =====
        if not recommendations.empty:
            st.markdown(f"Showing top {len(recommendations)} restaurants in **{city}** using **{method}**.")
            st.dataframe(
                recommendations[['name', 'city', 'cuisine', 'rating', 'rating_count', 'cost']].reset_index(drop=True)
            )

            csv = recommendations.to_csv(index=False)
            st.download_button(
                label="📥 Download Recommendations",
                data=csv,
                file_name="restaurant_recommendations.csv",
                mime="text/csv"
            )
        else:
            st.error("❌ No recommendations found.")
