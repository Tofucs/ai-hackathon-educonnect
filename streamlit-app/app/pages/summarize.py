# pages/summarize.py
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

st.title("Summary of User Messages")

if "messages" not in st.session_state or not st.session_state.messages:
    st.write("No messages to summarize yet.")
else:
    user_messages = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
    if not user_messages:
        st.write("No user messages to summarize yet.")
    else:
        st.write("### User Messages:")
        for i, msg in enumerate(user_messages, start=1):
            st.write(f"{i}. {msg}")
        if st.button("Summarize User Messages"):
            full_text = "\n".join(user_messages)
            try:
                response = client.chat.completions.create(
                    model="anthropic.claude-3.5-haiku",
                    messages=[
                        {"role": "system", "content": "You are an assistant that summarizes user messages concisely."},
                        {"role": "user", "content": f"Please provide a concise summary of the following user messages:\n\n{full_text}"}
                    ],
                )
                summary = response.choices[0].message.content
                st.subheader("Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"Error summarizing messages: {str(e)}")