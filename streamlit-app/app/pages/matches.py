# pages/matching.py
import os
import asyncio
import httpx
import openai
import nest_asyncio
import urllib.parse
import hashlib
import streamlit as st
from dotenv import load_dotenv
from aiohttp import ClientSession
from bs4 import BeautifulSoup

# Allow nested asyncio loops (useful when an event loop is already running)
nest_asyncio.apply()

# Load environment variables and set up OpenAI API
load_dotenv()
openai.api_base = os.getenv("OPENAI_BASE_URL")
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("Missing OpenAI API Key! Make sure it's set in your .env file.")
    st.stop()

st.set_page_config(page_title="Nonprofit Matches", layout="wide")
st.title("Nonprofit Matches (Swipe Interface)")

# SERP API key (consider storing this in an environment variable)
SERP_API_KEY = "1c92173370dc002dd7a0053eb47eeef87f9cce81bc61ccdcc766c1672a4b0087"

##########################################
# Helper Function for Rerunning the App
##########################################
def rerun_app():
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.write("Please refresh the page manually.")

##########################################
# Asynchronous Functions for SERP API Calls
##########################################
async def search_serpapi(prompt: str) -> list:
    # Clean the prompt: remove newlines and extra whitespace
    clean_query = prompt.replace("\n", " ")
    clean_query = " ".join(clean_query.split())
    # Append a nonprofit filter so that the results focus on nonprofit aid organizations
    filtered_query = f"{clean_query} nonprofit organization providing aid"
    # URL‑encode the query
    encoded_query = urllib.parse.quote(filtered_query)
    url = f"https://serpapi.com/search.json?q={encoded_query}&engine=google&api_key={SERP_API_KEY}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()
        results = [
            {
                "title": item.get("title", "No title"),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", "")
            }
            for item in data.get("organic_results", [])[:5]
        ]
        return results

async def summarize_with_ai(text: str, url: str) -> str:
    response = openai.chat.completions.create(
        model='anthropic.claude-3.5-sonnet.v2',
        messages=[
            {
                'role': 'user',
                'content': (
                    f"Summarize this snippet about a non-profit:\n\n{text}\n\n"
                    f"Include the context from this website: {url}. ONLY DISPLAY NONPROFIT ORGANIZATIONS PROVIDING AID, NOTHING ELSE. "
                    "If you cannot summarize, say no summary available."
                )
            },
        ],
    )
    summary = response.choices[0].message.content
    return summary.strip()

async def full_search_summarize(prompt: str) -> list:
    search_results = await search_serpapi(prompt)
    summaries = []
    for result in search_results:
        snippet_summary = await summarize_with_ai(result["snippet"], result["link"])
        summaries.append({
            "title": result.get("title", "No title"),
            "link": result.get("link", ""),
            "summary": snippet_summary
        })
    return summaries

##########################################
# Main Application Logic
##########################################

# Ensure that a summary has been generated from your Summary page.
if "latest_summary" not in st.session_state or not st.session_state.latest_summary:
    st.write("No summary found. Please generate a summary on the Summary page first.")
    st.stop()

# Initialize liked/disliked lists if they are not already in session state.
if "liked" not in st.session_state:
    st.session_state.liked = []
if "disliked" not in st.session_state:
    st.session_state.disliked = []

# Check if the summary has changed by comparing hashes.
current_summary = st.session_state.latest_summary
current_summary_hash = hashlib.sha256(current_summary.encode()).hexdigest()

# If a new summary is generated or nonprofit_matches hasn't been set yet, fetch new matches.
if st.session_state.get("summary_hash") != current_summary_hash or "nonprofit_matches" not in st.session_state:
    with st.spinner("Generating nonprofit matches..."):
        try:
            matches = asyncio.run(full_search_summarize(current_summary))
            # Filter out duplicates: remove any matches whose title is already in liked or disliked lists.
            liked_titles = {n.get("title", "").lower() for n in st.session_state.liked}
            disliked_titles = {n.get("title", "").lower() for n in st.session_state.disliked}
            new_matches = [m for m in matches if m.get("title", "").lower() not in liked_titles and m.get("title", "").lower() not in disliked_titles]
            st.session_state.nonprofit_matches = new_matches
            st.session_state.summary_hash = current_summary_hash
        except Exception as e:
            st.error(f"Error generating nonprofit matches: {e}")
            st.stop()
else:
    # Otherwise, filter the existing matches to remove any duplicates.
    filtered_matches = []
    liked_titles = {n.get("title", "").lower() for n in st.session_state.liked}
    disliked_titles = {n.get("title", "").lower() for n in st.session_state.disliked}
    for match in st.session_state.nonprofit_matches:
        title = match.get("title", "").lower()
        if title not in liked_titles and title not in disliked_titles:
            filtered_matches.append(match)
    st.session_state.nonprofit_matches = filtered_matches

##########################################
# Swipe Interface
##########################################

if st.session_state.nonprofit_matches:
    current = st.session_state.nonprofit_matches[0]
    
    st.markdown(f"## {current.get('title', 'No Title')}")
    st.write(current.get("summary", "No summary available."))
    st.write(f"[Visit Website]({current.get('link', '#')})")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("❌ Dislike"):
            st.session_state.disliked.append(current)
            st.session_state.nonprofit_matches.pop(0)
            rerun_app()
    with col2:
        if st.button("❤️ Like"):
            st.session_state.liked.append(current)
            st.session_state.nonprofit_matches.pop(0)
            rerun_app()
else:
    st.write("🎉 No more nonprofit matches available!")

##########################################
# Display Swipe History
##########################################

st.write("## Your Preferences")
st.write("### ❤️ Liked")
for item in st.session_state.liked:
    title = item.get("title") or item.get("Title") or "Unknown Title"
    st.write(f"- {title}")

st.write("### ❌ Disliked")
for item in st.session_state.disliked:
    title = item.get("title") or item.get("Title") or "Unknown Title"
    st.write(f"- {title}")