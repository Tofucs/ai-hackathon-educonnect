import streamlit as st
import os
# Set up the page configuration
st.set_page_config(page_title="Nonprofit New York", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .header-title {
        font-size: 70px;
        font-weight: bold;
        color: #FF4A4A;
        text-align: center;
        margin-bottom: 20px;
    }
    .description {
        font-size: 20px;
        color: #555555;
        text-align: left;
        margin-bottom: 30px;
    }

    .stButton button {
        background-color: #FF4A4A;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #D94040;
    }
    </style>
""", unsafe_allow_html=True)

image_path = os.path.join(os.getcwd(), "nonprofit.jpg")
st.image("./nonprofit.jpg", use_column_width=True)

# Centered Title
st.markdown('<h1 class="header-title">EduConnect.AI</h1>', unsafe_allow_html=True)

st.markdown('<h2 class="subheading">What is EduConnect.AI?</h1>', unsafe_allow_html=True)

# Centered Description
st.markdown('<p class="description">EduConnect.AI is an AI solution that helps connect underserviced students and school districts to educational non-profits to help bridge the education gap. </p>', unsafe_allow_html=True)

st.markdown('<h2 class="subheading">How do you use EduConnect.AI?</h1>', unsafe_allow_html=True)

# Centered Description
st.markdown('<p class="description">EduConnect.AI starts by asking the user a few preliminary questions before giving suggestions. The user then rates the suggestions EduConnect.AI has given. These ratings are then refined to give better and better suggestions so that you can find your ideal non-profit! Finally, EduConnect will help write a message to reach out to the non-profit to begin your connection. Get started by clicking the button below!</p>', unsafe_allow_html=True)


col1,col2,col3 = st.columns([1.4    ,1.,1.])
with col2:
    if st.button("Start Connecting!"):
        st.switch_page("pages/chat.py")  # Ensure this page exists in your Streamlit app
