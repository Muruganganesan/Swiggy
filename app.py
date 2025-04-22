import streamlit as st

st.set_page_config(page_title="Hotel Recommendation App", layout="wide")

import pandas as pd

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("swiggy_final.csv")
    return data

df = load_data()

# Header Section
st.title("🍽️ Restaurant Recommendation App")
st.markdown("We help you find the best restaurants that suit your taste and location.")

# Layout: Sidebar for filters
with st.sidebar:
    st.header("Filter Options")
    city = st.selectbox('Select City', sorted(df['city'].dropna().unique()))
    cuisine = st.multiselect('Select Cuisine(s)', sorted(df['cuisine'].dropna().unique()))
    min_rating = st.slider('Minimum Rating', min_value=0.0, max_value=5.0, value=3.0, step=0.1)

# Recommendation Engine
filtered_df = df[
    (df['city'] == city) &
    (df['rating'] >= min_rating)
]

if cuisine:
    filtered_df = filtered_df[filtered_df['cuisine'].isin(cuisine)]

# Main Content: Recommendations
st.subheader("Recommended Restaurants")

if not filtered_df.empty:
    for idx, row in filtered_df.iterrows():
        with st.expander(f"**{row['name']}** ({row['cuisine']})"):
            col1, col2 = st.columns([1, 3])
            # If you have images, display them:
            # col1.image(row['image_url'], width=100)
            col1.markdown(f"**Rating:** {row['rating']} ⭐️")
            col1.markdown(f"**Address:** {row['address']}")
            # Add more details if available
            col2.markdown(f"**Description:** {row.get('description', 'No description available.')}")
else:
    st.info("No restaurants found matching your criteria.")

# Optional: Add a chart for cuisine distribution
if not filtered_df.empty:
    st.markdown("### Cuisine Distribution")
    st.bar_chart(filtered_df['cuisine'].value_counts())

# Footer
st.markdown("---")
st.caption("Powered by Murugan | Data from Swiggy")
