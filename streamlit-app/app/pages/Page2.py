import streamlit as st
import random

import os

# Set the page config
st.set_page_config(page_title="Quokka Swipe", layout="centered")

# Initialize profiles in session state to persist changes
if "profiles" not in st.session_state:
    st.session_state.profiles = [
        {"name": "Quokkethoven", "description": "Composer of furry symphonies.", "image": "app/images/Bethoven.png"},
        {"name": "Quokkstra Stravinsky", "description": "Master of happy orchestras.", "image": "app/images/Stravinsky.png"},
        {"name": "Wolfgang Amadeus Quokka", "description": "A cheerful little maestro!", "image": "app/images/Wolfgang.png"},
        {"name": "Frederic Quokkopin", "description": "Nocturnes and sunshine.", "image": "app/images/Chopin.png"}
    ]

# Store swipe history in session state
if "swipe_history" not in st.session_state:
    st.session_state.swipe_history = []

# Store current profile to prevent repeats
if "current_profile" not in st.session_state:
    st.session_state.current_profile = None

# Select a new profile that is different from the last one
def get_new_profile():
    if not st.session_state.profiles:
        return None  # No more profiles to swipe
    new_profile = random.choice(st.session_state.profiles)
    
    # Ensure it's not the same as the last shown one
    while "current_profile" in st.session_state and st.session_state.current_profile and new_profile["name"] == st.session_state.current_profile["name"]:
        new_profile = random.choice(st.session_state.profiles)
    
    return new_profile

# Set the first profile if none exists
if st.session_state.current_profile is None and st.session_state.profiles:
    st.session_state.current_profile = get_new_profile()

# Display the profile
if st.session_state.current_profile:
    profile = st.session_state.current_profile
    st.image(profile["image"], use_container_width=True)
    st.markdown(f"### {profile['name']}")
    st.markdown(f"{profile['description']}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("❌ Nope"):
            st.session_state.swipe_history.append((profile["name"], "Nope"))
            st.session_state.profiles.remove(profile)  # Remove from available profiles
            st.session_state.current_profile = get_new_profile()  # Select a new profile
            st.rerun()
    with col2:
        if st.button("❤️ Like"):
            st.session_state.swipe_history.append((profile["name"], "Like"))
            st.session_state.profiles.remove(profile)  # Remove from available profiles
            st.session_state.current_profile = get_new_profile()  # Select a new profile
            st.rerun()

else:
    st.write("### No more profiles to swipe! 🐨")

# Show swipe history
if st.checkbox("Show swipe history"):
    st.write("### Swipe History")
    for name, decision in st.session_state.swipe_history:
        st.write(f"- {name}: {decision}")
