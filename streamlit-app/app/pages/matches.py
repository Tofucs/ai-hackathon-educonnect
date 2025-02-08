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

# Allow nested asyncio loops (useful for environments where an event loop is already running)
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

# SERP API key (consider loading this from an environment variable as well)
SERP_API_KEY = "1c92173370dc002dd7a0053eb47eeef87f9cce81bc61ccdcc766c1672a4b0087"

async def search_serpapi(prompt: str) -> list:
    query = prompt
    url = f"https://serpapi.com/search.json?q={query}&engine=google&api_key={SERP_API_KEY}"
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
                'content': f"Summarize this snippet about a non-profit:\n\n{text}\n\nInclude the context from this website: {url}"
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

# Streamlit UI: Enter a search query for nonprofits.
user_prompt = st.text_input("Enter your search query for nonprofits:", "financial aid educational non profit")

if st.button("Search Nonprofits"):
    with st.spinner("Loading nonprofit matches..."):
        try:
            results = asyncio.run(full_search_summarize(user_prompt))
        except Exception as e:
            st.error(f"Error during search: {e}")
            results = []
    
    if results:
        df = pd.DataFrame(results)
        st.dataframe(df)
    else:
        st.write("No nonprofit matches found.")
