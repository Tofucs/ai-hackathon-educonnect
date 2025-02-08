# pages/chat.py
import streamlit as st
import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_base = os.getenv("OPENAI_BASE_URL")
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("Missing OpenAI API Key! Make sure it's set in your .env file.")
    st.stop()

client = openai.OpenAI(api_key=openai.api_key)

st.title("Tell me about what you need!")

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are an AI assistant whose sole purpose is to match schools and students in need with nonprofit organizations that can help them. "
        "Your task is to ask only relevant questions to collect essential information from the school. "
        "Focus on asking for details such as the school's location, school size, and what specific kind of help they need. "
        "If the user provides irrelevant information or asks unrelated questions, kindly prompt them to provide the necessary details."
        "Only ask one question at a time to not overwhelm the user."
    )
}

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Before we start, can you tell me a little bit about your school?"}]

if st.button("Clear Chat History"):
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Before we start, can you tell me a little bit about your school?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    try:
        response = client.chat.completions.create(
            model="anthropic.claude-3.5-haiku",
            messages=[SYSTEM_MESSAGE] + st.session_state.messages,
        )
        assistant_reply = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant"):
            st.write(assistant_reply)
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")