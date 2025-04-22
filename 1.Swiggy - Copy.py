import streamlit as st
import pandas as pd

# Page config
st.set_page_config(page_title="🍽️ Restaurant Recommendation App", layout="wide")

# Load dataset with caching
@st.cache_data
def load_data():
    data = pd.read_csv(r'C:\Users\admin\Music\Guvi\Swiggy\swiggy_final.csv')
    # Ensure rating is float for slider filtering
    data['rating'] = pd.to_numeric(data['rating'], errors='coerce').fillna(0)
    return data

df = load_data()

# Sidebar - Filters
with st.sidebar:
    st.title("🔍 Find Your Restaurant")
    st.markdown("Filter restaurants by city, cuisine, and rating.")

    city = st.selectbox(
        'Select City',
        options=sorted(df['city'].dropna().unique()),
        help="Choose your city to find restaurants near you."
    )

    cuisine = st.multiselect(
        'Select Cuisine(s)',
        options=sorted(df['cuisine'].dropna().unique()),
        help="Pick one or more cuisines you like."
    )

    min_rating = st.slider(
        'Minimum Rating',
        min_value=0.0,
        max_value=5.0,
        value=3.0,
        step=0.1,
        help="Filter restaurants with rating equal or above this value."
    )

    # Optional: Add "Open Now" filter if data available
    # open_now = st.checkbox("Open Now")

# Main Page Title & Instructions
st.title("🍽️ Restaurant Recommendation App")
st.markdown(
    """
    Welcome! Use the filters on the left to find the best restaurants that suit your taste and location.
    \n
    
    """
)

# Filter DataFrame based on selections
filtered_df = df[
    (df['city'] == city) &
    (df['rating'] >= min_rating)
]

if cuisine:
    filtered_df = filtered_df[filtered_df['cuisine'].isin(cuisine)]

# Show number of results found
st.markdown(f"### 🍴 Found {len(filtered_df)} restaurants matching your criteria")

# Recommendations Section
if not filtered_df.empty:
    for idx, row in filtered_df.iterrows():
        with st.expander(f"**{row['name']}** ({row['cuisine']})"):
            col1, col2 = st.columns([1, 3])

            # Display image if available
            if 'image_url' in row and pd.notna(row['image_url']) and row['image_url'].strip() != '':
                col1.image(row['image_url'], width=120)
            else:
                col1.markdown("🖼️ No image available")

            # Restaurant details
            col1.markdown(f"**Rating:** {row['rating']} ⭐️")
            col1.markdown(f"**Address:** {row['address']}")

            # Description or placeholder
            description = row.get('description', 'No description available.')
            col2.markdown(f"**Description:** {description}")

            # Optional: Add a button for user feedback (non-functional placeholder)
            if col2.button("Add Your Review", key=f"review_{idx}"):
                st.info("Thank you for your interest! Review feature coming soon.")

else:
    st.info("😞 No restaurants found matching your criteria. Try adjusting the filters.")

# Cuisine Distribution Chart
if not filtered_df.empty:
    st.markdown("---")
    st.markdown("### 📊 Cuisine Distribution in Results")
    cuisine_counts = filtered_df['cuisine'].value_counts()
    st.bar_chart(cuisine_counts)

# Footer
st.markdown("---")
st.caption("Powered by Data from Swiggy")

#"C:\Users\admin\Music\Guvi\Swiggy\1.Swiggy.py"