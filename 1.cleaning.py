import pandas as pd

# 1. Dataset ஐ Load செய்யவும்
df = pd.read_csv(r"C:\Users\admin\Music\Guvi\Swiggy\swiggy full data.csv") 

# 2. Remove duplicate rows
df = df.drop_duplicates()

# 3. Remove rows where rating is '--'
df = df[df['rating'] != '--']

# 4. Convert 'rating' column to float
df['rating'] = df['rating'].astype(float)

# 5. Clean 'rating_count' column

# Convert "1.2K" to "1200", "3K" to "3000"
def convert_rating_count(val):
    if isinstance(val, str):
        val = val.replace('+', '').replace('ratings', '').replace(',', '').strip()
        if 'K' in val:
            val = val.replace('K', '')
            try:
                return float(val) * 1000
            except:
                return None
        else:
            try:
                return float(val)
            except:
                return None
    return val

df['rating_count'] = df['rating_count'].apply(convert_rating_count)

# 6. Drop rows with any missing values
df = df.dropna()

# 7. Drop 'lic_no' column if present
if 'lic_no' in df.columns:
    df = df.drop(columns=['lic_no'])
    print("'lic_no' column removed.")

# 8. Save cleaned data
df.to_csv("cleaned_data.csv", index=False)
print("✅ Cleaned data saved as 'cleaned_data.csv'")


