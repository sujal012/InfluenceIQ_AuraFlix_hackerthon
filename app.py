import streamlit as st  # Import Streamlit first
st.set_page_config(page_title="Influencer Recommendation", layout="wide")  # Set config at the top

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Function to convert string numbers with 'M' and 'K' to float
def convert_to_number(value):
    if isinstance(value, str):
        value = value.replace(',', '')  # Remove commas
        if "M" in value:
            return float(value.replace("M", "")) * 1e6
        elif "K" in value:
            return float(value.replace("K", "")) * 1e3
    return float(value)

# Load influencer dataset
@st.cache_data
def load_data():
    influencer_df = pd.read_csv("influencers.csv")

    # Convert follower and view counts properly
    influencer_df['followers'] = influencer_df['followers'].apply(convert_to_number)
    influencer_df['Average views'] = influencer_df['Average views'].apply(convert_to_number)

    # Calculate engagement rate
    influencer_df['engagement_rate'] = influencer_df['Average views'] / influencer_df['followers']

    return influencer_df

influencer_df = load_data()

# Preprocessing function
def preprocess_data(df):
    df = df.dropna()  # Remove missing values
    scaler = StandardScaler()
    numeric_cols = ['followers', 'engagement_rate']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

influencer_df = preprocess_data(influencer_df)

# Recommendation function
def recommend_influencers(user_input, df, top_n=5):
    user_vector = np.array(user_input).reshape(1, -1)
    influencer_vectors = df[['followers', 'engagement_rate']].values
    similarities = cosine_similarity(user_vector, influencer_vectors)[0]
    df['similarity'] = similarities
    top_influencers = df.sort_values(by='similarity', ascending=False).head(top_n)
    return top_influencers

# --- Streamlit UI ---
st.sidebar.title("📊 Data Insights")
st.sidebar.write("Total Influencers: ", len(influencer_df))

# Followers distribution
fig, ax = plt.subplots(figsize=(5, 3))
sns.histplot(influencer_df['followers'], bins=30, kde=True, ax=ax)
ax.set_title("Followers Distribution")
st.sidebar.pyplot(fig)

# Engagement Rate Pie Chart
fig, ax = plt.subplots()
labels = ['Low Engagement (<0.2)', 'Medium (0.2 - 0.6)', 'High (>0.6)']
sizes = [
    len(influencer_df[influencer_df['engagement_rate'] < 0.2]),
    len(influencer_df[(influencer_df['engagement_rate'] >= 0.2) & (influencer_df['engagement_rate'] <= 0.6)]),
    len(influencer_df[influencer_df['engagement_rate'] > 0.6])
]
ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
ax.set_title("Engagement Rate Distribution")
st.sidebar.pyplot(fig)

# --- Main UI ---
st.title("🎯 Influencer Recommendation System")
st.write("Find the best influencers for your campaign based on followers & engagement rate!")

# User Input Section
col1, col2 = st.columns(2)
with col1:
    followers = st.number_input("Enter required followers count:", min_value=1000, step=1000)
with col2:
    engagement_rate = st.slider("Select engagement rate:", 0.0, 1.0, 0.05)

# Generate Recommendations
if st.button("🔍 Find Influencers"):
    user_input = [followers, engagement_rate]
    recommendations = recommend_influencers(user_input, influencer_df)

    st.write("### 🎖 Recommended Influencers:")
    st.dataframe(recommendations[['username', 'followers', 'engagement_rate', 'similarity']])

    # Bar Chart for Recommended Influencers
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=recommendations, x='username', y='followers', palette="coolwarm", ax=ax)
    ax.set_title("Top Recommended Influencers by Followers")
    plt.xticks(rotation=30)
    st.pyplot(fig)

st.write("🚀 **Built for Auraflix Hackathon! **")
