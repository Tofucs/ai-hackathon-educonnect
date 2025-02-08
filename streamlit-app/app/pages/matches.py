# pages/matches.py
import os
import asyncio
import httpx
import openai
from aiohttp import ClientSession
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import nest_asyncio
import urllib.parse

# Allow nested asyncio loops (useful in environments where an event loop is already running)
nest_asyncio.apply()

# Load environment variables and set up OpenAI API
load_dotenv()
openai.api_base = os.getenv("OPENAI_BASE_URL")
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("Missing OpenAI API Key! Make sure it's set in your .env file.")
    st.stop()

# Set up Streamlit page configuration
st.set_page_config(page_title="Nonprofit Matches", layout="wide")
st.title("Nonprofit Matches")

# SERP API key (consider loading this from an environment variable)
SERP_API_KEY = "1c92173370dc002dd7a0053eb47eeef87f9cce81bc61ccdcc766c1672a4b0087"

async def search_serpapi(prompt: str) -> list:
    # Remove newline characters and extra whitespace from the summary
    clean_query = prompt.replace("\n", " ")
    clean_query = " ".join(clean_query.split())
    # Append a nonprofit filter to the query to ensure results focus on nonprofit aid organizations
    filtered_query = f"{clean_query} nonprofit organization providing aid"
    # URL-encode the cleaned and filtered query
    encoded_query = urllib.parse.quote(filtered_query)
    url = f"https://serpapi.com/search.json?q={encoded_query}&engine=google&api_key={SERP_API_KEY}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()
        results = [
            {
                "title": item["title"],
                "link": item["link"],
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

async def fetch_full_page(url: str) -> str:
    async with ClientSession() as session:
        async with session.get(url) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs if len(p.get_text()) > 50])
            return text[:3000].strip()

async def full_search_summarize(prompt: str) -> list:
    search_results = await search_serpapi(prompt)
    summaries = []
    for result in search_results:
        snippet_summary = await summarize_with_ai(result["snippet"], result["link"])
        summaries.append({
            "Title": result["title"],
            "Link": result["link"],
            "Summary": snippet_summary
        })
    return summaries

# Check if a summary exists in session state
if "latest_summary" not in st.session_state or not st.session_state.latest_summary:
    st.write("No summary found. Please generate a summary in the Summary page first.")
    st.stop()
else:
    # Use the stored summary from summarize.py as the query for the search
    query = st.session_state.latest_summary
    
    with st.spinner("Loading nonprofit matches..."):
        try:
            results = asyncio.run(full_search_summarize(query))
        except Exception as e:
            st.error(f"Error during search: {e}")
            results = []
    
    if results:
        df = pd.DataFrame(results)
        st.dataframe(df)
    else:
        st.write("No nonprofit matches found.")