import streamlit as st
import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OpenAI API Key! Make sure it's set in your .env file.")
    st.stop()

client = openai.OpenAI(api_key=api_key)

st.title("AI Chatbot")

# Define AI's context with a system message
SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are an AI assistant designed to match students and schools who are low-income or in need by matching them with nonprofit organizations"
               "You will ask questions about school location, size, what sort of help they need, etc."
               "You will ask these questions one by one to not overwhelm the user"
               "If the user asks irrelevant questions or responses then tell them you do not have a response to that, and prompt to tell you more about their needs."
}

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [] 
    initial_message = "Hello! Before we start, can you tell me a little bit about your school?"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input using chat input (always stays at the bottom)
user_input = st.chat_input("Type your message...")
if user_input:
    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.write(user_input)

    # Generate response from OpenAI
    try:
        response = client.chat.completions.create(
            model="anthropic.claude-3.5-haiku",  # Change model if needed
            messages=st.session_state.messages,
        )
        assistant_reply = response.choices[0].message.content

        # Append assistant's reply to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

        with st.chat_message("assistant"):
            st.write(assistant_reply)

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")