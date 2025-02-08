import streamlit as st
import os
# Set up the page configuration
st.set_page_config(page_title="EduConnect.AI", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    @keyframes fadeIn {
      0% {
        opacity: 0;
      }
      100% {
        opacity: 1;
      }
    }
    img {
        border-radius: 20px;
    }
    .header-title {
        font-size: 80px !important;
        font-weight: bold;
        color: #FF4A4A;
        text-align: center;
        margin-bottom: 20px;
    }
    .description {
        font-size: 21px !important;
        color: #555555;
        text-align: center;
        margin-bottom: 10px;
        margin: 0 100px 10px 100px !important;
        opacity: 0;
        
        animation: fadeIn 2s ease-in forwards;
    }
    .subheading {
        text-align: center;
        opacity: 0;
        transform: translateY(20px);
        animation: fadeIn 2.5s ease-in forwards;
        margin-bottom: 10px !important;
     
    }
    .stButton button {
        margin-top: 30px;
        background-color: #FF4A4A;
        color: white;
        border-radius: 10px;
        padding: 22px 40px;
        font-size: 30px !important;
        font-weight: bold;
        min-width: 200px;
        white-space: nowrap;
        border: none;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #D94040;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])  # Adjust proportions as needed
with col2: 
    st.image("./app/images/EduConnect.png", use_container_width=True)

# Centered Title
#st.markdown('<h1 class="header-title">EduConnect.AI</h1>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])  # Adjust proportions as needed

with col2:  # Center the image in the second column
    st.image("./app/images/nonprofit.jpg", use_container_width=True)

st.markdown('<h2 class="subheading">What is EduConnect.AI?</h1>', unsafe_allow_html=True)

# Centered Description
st.markdown('<p class="description">EduConnect.AI is an AI solution that helps connect underserviced students and school districts to educational non-profits to help bridge the education gap. </p>', unsafe_allow_html=True)

st.markdown('<h2 class="subheading">How do you use EduConnect.AI?</h1>', unsafe_allow_html=True)

# Centered Description
st.markdown('<p class="description">EduConnect.AI begins by asking you a few initial questions to tailor its suggestions. You can rate the recommendations provided, and these ratings help refine our AI agent to deliver increasingly accurate suggestions, guiding you to your ideal non-profit. Once you find the perfect match, EduConnect will assist in drafting a message to help you connect with the organization. Ready to start? Click the button below!</p>', unsafe_allow_html=True)

col1,col2,col3 = st.columns([1.2,1.,1.])
with col2:
    if st.button("Start Connecting!"):
        st.switch_page("pages/chat.py")  # Ensure this page exists in your Streamlit app
