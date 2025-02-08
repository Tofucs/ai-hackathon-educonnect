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
st.set_page_config(page_title="EduConnect.AI", layout="wide")
st.title("Summary of Requests")

if "messages" not in st.session_state or not st.session_state.messages:
    st.write("No messages to summarize yet.")
else:
    # Build conversation text including both the assistant's prompts and the user's responses.
    conversation_text = ""
    for msg in st.session_state.messages:
        if msg["role"] in ["assistant", "user"]:
            role_label = "Assistant" if msg["role"] == "assistant" else "User"
            conversation_text += f"{role_label}: {msg['content']}\n"
    
    placeholder = st.empty()
    placeholder.write("Loading summary...")

    try:
        # System prompt instructs the AI to include externally verified information when available.
        SYSTEM_PROMPT = (
            "You are an AI assistant whose sole purpose is to match schools and students in need with nonprofit organizations that can help them. "
            "Your task is to ask only relevant questions to collect essential information from the school, including details such as the school's location, "
            "school size, and the specific kind of help needed. "
            "If the conversation contains any externally verified information about a high school (e.g., from online searches, such as its location or size), "
            "include that information in your summary. "
            "Now, based on the following conversation—which includes both your prompts and the user's responses—provide a concise summary that captures the essential details needed for matching. "
            "You should not ask the user to provide more details, only summarize the details they have provided."
        )
        
        # Generate the summary.
        summary_response = client.chat.completions.create(
            model="anthropic.claude-3.5-haiku",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": conversation_text}
            ],
        )
        summary = summary_response.choices[0].message.content.strip()
        
        # Now, create a search query based on the summary.
        # The query must focus on details like location and needs while excluding any school-related info.
        QUERY_PROMPT = (
            "Based on the following summary, generate a search query that focuses on nonprofit organizations providing aid. "
            "The search query should emphasize details such as location, specific needs, and types of assistance, "
            "but it must NOT include any information about the school or its website. "
            "Only include keywords relevant to nonprofit organizations and their services. "
            "The query should only contain the support needed and the location, NOTHING ELSE. it should be a couple words or phrases at most. make sure the last word is always nonprofit "
            "\n\nSummary:\n" + summary
        )
        
        # Provide both a system and a user message to meet the model's requirements.
        query_response = client.chat.completions.create(
            model="anthropic.claude-3.5-haiku",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": QUERY_PROMPT}
            ],
        )
        search_query = query_response.choices[0].message.content.strip()
        
        placeholder.empty()
        st.write("**Summary:**")
        st.write(summary)
        st.write("**Search Query for Nonprofit Organizations Providing Aid:**")
        st.write(search_query)
        
        # Store the generated search query in session state for use in matches.py.
        st.session_state.latest_summary = search_query
        
    except Exception as e:
        placeholder.empty()
        st.error(f"Error summarizing conversation: {str(e)}")