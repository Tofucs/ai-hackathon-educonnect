import streamlit as st
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
import boto3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import time

load_dotenv()
openai.api_base = os.getenv("OPENAI_BASE_URL")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Sample nonprofits dataset
nonprofits = [
    {"name": "Green Earth", "description": "Fighting climate change through tree planting and conservation efforts."},
    {"name": "Tech for Kids", "description": "Providing underprivileged children with access to technology and coding education."},
    {"name": "Animal Rescue League", "description": "Rescuing and rehabilitating abandoned and abused animals."},
    {"name": "Food for All", "description": "Delivering nutritious meals to homeless and food-insecure individuals."},
    {"name": "Women in STEM", "description": "Empowering women in science, technology, engineering, and mathematics."},
    {"name": "Blue Earth", "description": "Fighting climate change through tree planting and conservation efforts."},
    {"name": "Red Earth", "description": "Fighting climate change through tree planting and conservation efforts."},
    {"name": "Women in REM", "description": "Empowering women in science, technology, engineering, and mathematics."},
    {"name": "Women in RAM", "description": "Empowering women in science, technology, engineering, and mathematics."},
]

# Initialize session state
if "liked" not in st.session_state:
    st.session_state.liked = []
if "disliked" not in st.session_state:
    st.session_state.disliked = []
if "recommendations" not in st.session_state:
    st.session_state.recommendations = nonprofits  # Start with all nonprofits


# Extract top keywords
def extract_keywords(text, top_n=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    top_keywords = feature_array[tfidf_sorting][:top_n]
    
    #st.write(f"**Keywords for:** {text}")
    #st.write(top_keywords.tolist())
    
    return " ".join(top_keywords)  # Convert keywords back into text for embeddings

# Function to get text embeddings
def get_embedding(text):
    keywords = extract_keywords(text)
    
    response = openai.embeddings.create(
        input=keywords,
        model="amazon.titan-text-embeddings.v2"
    )
    return np.array(response.data[0].embedding)

# Function to update recommendations
def update_recommendations():
    if not st.session_state.liked:
        st.session_state.recommendations = nonprofits  # Show all if no preferences
        return

    # Get embeddings for liked and disliked nonprofits
    liked_embeddings = [get_embedding(nonprofit["description"]) for nonprofit in st.session_state.liked]
    disliked_embeddings = [get_embedding(nonprofit["description"]) for nonprofit in st.session_state.disliked]
    
    new_recommendations = []
    
    for nonprofit in nonprofits:
        if nonprofit in st.session_state.liked or nonprofit in st.session_state.disliked:
            continue  # Skip already seen

        nonprofit_embedding = get_embedding(nonprofit["description"])

        # Compute similarity with liked and disliked nonprofits
        liked_similarities = [cosine_similarity([nonprofit_embedding], [le])[0][0] for le in liked_embeddings]
        disliked_similarities = [cosine_similarity([nonprofit_embedding], [de])[0][0] for de in disliked_embeddings] if disliked_embeddings else [0]

        avg_liked_similarity = np.mean(liked_similarities) if liked_similarities else 0
        avg_disliked_similarity = np.mean(disliked_similarities) if disliked_similarities else 0

        # Keep nonprofits more similar to liked ones and dissimilar to disliked ones
        if avg_liked_similarity > avg_disliked_similarity:
            new_recommendations.append((nonprofit, avg_liked_similarity))

    # Sort recommendations by similarity score
    new_recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Update state
    st.session_state.recommendations = [x[0] for x in new_recommendations]

def update_recommendations2():
    if not st.session_state.liked and not st.session_state.disliked:
        st.session_state.recommendations = nonprofits  # Show all if no preferences
        return

    # Ensure disliked nonprofits are completely excluded from recommendations
    remaining_nonprofits = [n for n in nonprofits if n not in st.session_state.disliked and n not in st.session_state.liked]

    # Get embeddings for liked and disliked nonprofits
    liked_embeddings = [get_embedding(nonprofit["description"]) for nonprofit in st.session_state.liked] if st.session_state.liked else []
    disliked_embeddings = [get_embedding(nonprofit["description"]) for nonprofit in st.session_state.disliked] if st.session_state.disliked else []

    new_recommendations = []

    for nonprofit in remaining_nonprofits:

        nonprofit_embedding = get_embedding(nonprofit["description"])

        # Compute similarity with liked and disliked nonprofits
        liked_similarities = [cosine_similarity([nonprofit_embedding], [le])[0][0] for le in liked_embeddings] if liked_embeddings else [0]
        disliked_similarities = [cosine_similarity([nonprofit_embedding], [de])[0][0] for de in disliked_embeddings] if disliked_embeddings else [0]

        avg_liked_similarity = np.mean(liked_similarities) if liked_similarities else 0
        avg_disliked_similarity = np.mean(disliked_similarities) if disliked_similarities else 0

        # Remove nonprofits that are too similar to disliked ones
        if avg_disliked_similarity >= 0.5:  # Adjust threshold if needed
            continue

        # Add nonprofits with a higher similarity to liked ones
        new_recommendations.append((nonprofit, avg_liked_similarity))

    # Sort recommendations by similarity score
    new_recommendations.sort(key=lambda x: x[1], reverse=True)

    # Update state
    st.session_state.recommendations = [x[0] for x in new_recommendations]



# Streamlit UI
st.title("Nonprofit Swipe App")

if st.session_state.recommendations:
    current = st.session_state.recommendations[0]  # Show first recommendation

    st.markdown("""
        <style>
        .recommendation-box {
            border: 2px solid #000;
            border-radius: 12px;
            padding: 20px;

            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="recommendation-box">
            <h2>{current["name"]}</h2>
            <p>{current["description"]}</p>
        </div>
    """, unsafe_allow_html=True)
    #st.subheader(current["name"])
    #st.write(current["description"])

    col1, col2 = st.columns(2)

    with col1:
        if st.button("❌ Dislike"):
            st.session_state.disliked.append(current)
            update_recommendations2()
            st.rerun()

    with col2:
        if st.button("❤️ Like"):
            st.session_state.liked.append(current)
            update_recommendations2()
            st.rerun()

else:
    st.write("🎉 No more recommendations available!")

# Show swipe history
st.write("## Your Preferences")
st.write("### ❤️ Liked")
for item in st.session_state.liked:
    st.write(f"- {item['name']}")

st.write("### ❌ Disliked")
for item in st.session_state.disliked:
    st.write(f"- {item['name']}")
