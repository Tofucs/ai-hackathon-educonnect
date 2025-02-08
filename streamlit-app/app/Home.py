import streamlit as st

# Set up the page configuration
st.set_page_config(page_title="Nonprofit New York", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .header-title {
        font-size: 60px;
        font-weight: bold;
        color: #FF4A4A;
    }
    .description {
        font-size: 20px;
        color: #555555;
    }
    .button-container {
        display: flex;
        gap: 5px;
        justify-content: flex-start;
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
st.markdown('<h1 class="header-title">EduConnect.AI</h1>', unsafe_allow_html=True)

# Button Layout
st.markdown('<div class="button-container">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    if st.button("Home"):
        st.switch_page("Home.py")
with col2:
    if st.button("Chat"):
        st.switch_page("pages/chat.py")
with col3:
    st.button("Selection")
with col4:
    if st.button("Summary"):
        st.switch_page("pages/summarize.py")
st.markdown('</div>', unsafe_allow_html=True)


# Main Section
st.markdown('<p class="description">Nonprofit New York champions and strengthens nonprofits through capacity building and advocacy to cultivate a unified, just, and powerful sector.</p>', unsafe_allow_html=True)
