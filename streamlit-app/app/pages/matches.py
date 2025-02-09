# pages/matches.py
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
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Initialize necessary session state variables if they don't exist.
if "liked" not in st.session_state:
    st.session_state.liked = []
if "disliked" not in st.session_state:
    st.session_state.disliked = []
if "nonprofit_matches" not in st.session_state:
    st.session_state.nonprofit_matches = []
if "messages" not in st.session_state:
    st.session_state.messages = []
# Note: "latest_summary" and "summary_hash" may or may not be set already.

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
# Generate Summary (if needed)
##########################################
# If there is no generated search query (stored in st.session_state.latest_summary),
# then try to generate one from conversation messages.
if "latest_summary" not in st.session_state or not st.session_state.latest_summary:
    if st.session_state.messages:
        # Build conversation text from messages.
        conversation_text = ""
        for msg in st.session_state.messages:
            if msg["role"] in ["assistant", "user"]:
                role_label = "Assistant" if msg["role"] == "assistant" else "User"
                conversation_text += f"{role_label}: {msg['content']}\n"
        with st.spinner("Generating summary and search query..."):
            try:
                # System prompt to generate the summary.
                SYSTEM_PROMPT = (
                    "You are an AI assistant whose sole purpose is to match schools and students in need with nonprofit organizations that can help them. "
                    "Your task is to ask only relevant questions to collect essential information from the school, including details such as the school's location, "
                    "school size, and the specific kind of help needed. "
                    "If the conversation contains any externally verified information about a high school (e.g., from online searches, such as its location or size), "
                    "include that information in your summary. "
                    "Now, based on the following conversation—which includes both your prompts and the user's responses—provide a concise summary that captures the essential details needed for matching. "
                    "You should not ask the user to provide more details, only summarize the details they have provided."
                )
                summary_response = openai.chat.completions.create(
                    model="anthropic.claude-3.5-haiku",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": conversation_text}
                    ],
                )
                summary = summary_response.choices[0].message.content.strip()
                
                # Now, generate a search query based on the summary.
                QUERY_PROMPT = (
                    "Based on the following summary, generate a search query that focuses on nonprofit organizations providing aid. "
                    "The search query should emphasize details such as location, specific needs, and types of assistance, "
                    "but it must NOT include any information about the school or its website. "
                    "Only include keywords relevant to nonprofit organizations and their services. "
                    "The query should only contain the support needed and the location, NOTHING ELSE. It should be a couple words or phrases at most. "
                    "Make sure the last word is always 'nonprofit'."
                    "\n\nSummary:\n" + summary
                )
                query_response = openai.chat.completions.create(
                    model="anthropic.claude-3.5-haiku",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": QUERY_PROMPT}
                    ],
                )
                search_query = query_response.choices[0].message.content.strip()
                st.session_state.latest_summary = search_query
            except Exception as e:
                st.error(f"Error generating summary/search query: {e}")
                st.stop()
    else:
        st.write("No conversation messages found to generate a summary. Please provide conversation messages first.")
        st.stop()










##########################################
# Asynchronous Functions for SERP API Calls
##########################################

async def search_serpapi(prompt: str) -> list:
    """
    Perform a single SERP API search request for the given prompt.
    """
    query = prompt
    url = f"https://serpapi.com/search.json?q={query}&engine=google&api_key={SERP_API_KEY}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()
        results = [
            {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", "")
            }
            for item in data.get("organic_results", [])
        ]
    return results

async def is_nonprofit(text: str, url: str) -> bool:
    """
    Determine—via an AI call—whether the given snippet (or full page if snippet is empty)
    suggests that the website represents a nonprofit organization.
    The AI is instructed to respond with a simple 'Yes' or 'No'.
    """
    
    response = openai.chat.completions.create(
        model='anthropic.claude-3.5-sonnet.v2',
        messages=[
            {
                'role': 'user',
                'content': (
                    f"Based on the following snippet and URL, determine if the website likely represents a nonprofit organization. "
                    f"Do not attempt to read from the url website. Instead just make a guess from the webpage name"
                    f"You must be VERY confident"
                    f"Answer only 'Yes' or 'No'.\n\nSnippet: {text}\n\nURL: {url}"
                )
            },
        ],
    )
    '''
    if not (answer.startswith("yes") or answer.startswith("no")):
        additional_text = await fetch_full_page(url)
        # Reuse the original prompt plus additional scraped content.
        prompt_text_extended = (
            f"I previously asked: Based on the following snippet and URL, determine if the website likely represents a nonprofit organization. "
            f"Answer only 'Yes' or 'No'.\n\nSnippet: {text}\n\nURL: {url}\n\n"
            f"I have also scraped additional context from the website: {additional_text}\n\n"
            f"Please answer with only 'Yes' or 'No'."
        )
        response = openai.chat.completions.create(
            model='anthropic.claude-3.5-sonnet.v2',
            messages=[{'role': 'user', 'content': prompt_text_extended}],
        )
        answer = response.choices[0].message.content.strip().lower()'''
    answer = response.choices[0].message.content.strip().lower()
    return answer.startswith("yes")

async def retrieve_nonprofit_candidates(prompt: str, max_candidates: int = 5, max_results: int = 100) -> list:
    """
    Use SERP API (with pagination) to search for the prompt.
    Only return candidate results that have '.org' in their URL and for which an AI (via is_nonprofit)
    judges as likely a nonprofit.
    Stop when either max_candidates have been collected or when max_results have been processed.
    """
    candidates = []
    total_processed = 0
    start = 0

    while total_processed < max_results and len(candidates) < max_candidates:
        url = f"https://serpapi.com/search.json?q={prompt}&engine=google&api_key={SERP_API_KEY}&start={start}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            data = response.json()
        organic_results = data.get("organic_results", [])
        if not organic_results:
            break

        for item in organic_results:
            if total_processed >= max_results:
                break
            total_processed += 1
            link = item.get("link", "")
            # Only consider URLs with .org
            if ".org" not in link:
                continue
            snippet = item.get("snippet", "")
            if await is_nonprofit(snippet, link):
                candidates.append({
                    "title": item.get("title", ""),
                    "link": link,
                    "snippet": snippet
                })
                if len(candidates) >= max_candidates:
                    break
        start += len(organic_results)
    return candidates

async def summarize_with_ai(text: str, url: str) -> str:
    """
    Generate a summary of the snippet with context from the given URL.
    """
    response = openai.chat.completions.create(
        model='anthropic.claude-3.5-sonnet.v2',
        messages=[
            {
                'role': 'user',
                'content': f"Summarize this snippet about a non-profit:\n\n{text}\n\nInclude any context you know about {url}"
                f"If unable to gather enough significant information, please say 'No summary available"
                f"If at least two facts are present about the non-profit, please attempt to summarize."
            },
        ],
    )
    summary = response.choices[0].message.content.strip()
    return summary

async def generate_nonprofit_summaries(candidates: list) -> list:
    """
    For each candidate result, generate a summary.
    """
    summaries = []
    for candidate in candidates:
        snippet_summary = await summarize_with_ai(candidate["snippet"], candidate["link"])
        summaries.append({
            "title": candidate["title"],
            "link": candidate["link"],
            "summary": snippet_summary
        })
    return summaries

async def full_search_summarize(prompt: str) -> list:
    """
    Full RAG pipeline: First retrieve candidate nonprofit websites,
    then generate summaries for each candidate.
    """
    candidates = await retrieve_nonprofit_candidates(prompt)
    return await generate_nonprofit_summaries(candidates)















##########################################
# Functions for Recommendation Algorithm
##########################################
def extract_keywords(text, top_n=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    top_keywords = feature_array[tfidf_sorting][:top_n]
    return " ".join(top_keywords)

def get_embedding(text):
    # Extract keywords from the text to focus the embedding.
    keywords = extract_keywords(text)
    response = openai.embeddings.create(
        input=keywords,
        model="amazon.titan-text-embeddings.v2"
    )
    return np.array(response.data[0].embedding)

def update_recommendations():
    """
    Update the list of nonprofit matches based on the cosine similarity
    between the embeddings of each candidate's summary and those of the liked
    and disliked nonprofits. Candidates that are too similar to disliked items
    (threshold >= 0.5) are filtered out.
    """
    # If no preferences have been set, do nothing.
    if not st.session_state.liked and not st.session_state.disliked:
        return

    new_recommendations = []

    # Iterate only over candidates that haven't been swiped on yet.
    for candidate in st.session_state.nonprofit_matches:
        if candidate in st.session_state.liked or candidate in st.session_state.disliked:
            continue

        # Use the "summary" field from your SERP API results.
        candidate_embedding = get_embedding(candidate["summary"])

        # Get embeddings for liked and disliked nonprofits (using their "summary")
        liked_embeddings = (
            [get_embedding(n["summary"]) for n in st.session_state.liked]
            if st.session_state.liked else []
        )
        disliked_embeddings = (
            [get_embedding(n["summary"]) for n in st.session_state.disliked]
            if st.session_state.disliked else []
        )

        # Compute cosine similarities.
        liked_similarities = (
            [cosine_similarity([candidate_embedding], [emb])[0][0] for emb in liked_embeddings]
            if liked_embeddings else [0]
        )
        disliked_similarities = (
            [cosine_similarity([candidate_embedding], [emb])[0][0] for emb in disliked_embeddings]
            if disliked_embeddings else [0]
        )

        avg_liked_similarity = np.mean(liked_similarities) if liked_similarities else 0

        # Apply a penalty: if any disliked similarity is high, mark candidate for exclusion.
        penalty = 0
        for d_sim in disliked_similarities:
            if d_sim >= 0.5:
                penalty = 1
                break

        # Only add candidates with no penalty.
        if penalty == 0:
            new_recommendations.append((candidate, avg_liked_similarity))

    # Sort the remaining candidates by their average liked similarity (highest first).
    new_recommendations.sort(key=lambda x: x[1], reverse=True)
    st.session_state.nonprofit_matches = [x[0] for x in new_recommendations]

##########################################
# Main Application Logic
##########################################
# At this point, st.session_state.latest_summary should be set (either previously or just generated).
current_summary = st.session_state.latest_summary
current_summary_hash = hashlib.sha256(current_summary.encode()).hexdigest()

# If a new summary is generated (or if nonprofit_matches is not set), fetch new matches.
if st.session_state.get("summary_hash") != current_summary_hash or "nonprofit_matches" not in st.session_state or not st.session_state.nonprofit_matches:
    with st.spinner("Generating nonprofit matches..."):
        try:
            matches = asyncio.run(full_search_summarize(current_summary))
            # Filter out any matches whose title (in lowercase) is already in liked or disliked lists.
            liked_titles = {n.get("title", "").lower() for n in st.session_state.liked}
            disliked_titles = {n.get("title", "").lower() for n in st.session_state.disliked}
            new_matches = [m for m in matches if m.get("title", "").lower() not in liked_titles and m.get("title", "").lower() not in disliked_titles]
            st.session_state.nonprofit_matches = new_matches
            st.session_state.summary_hash = current_summary_hash
        except Exception as e:
            st.error(f"Error generating nonprofit matches: {e}")
            st.stop()
else:
    # Otherwise, filter existing matches to remove duplicates.
    filtered_matches = []
    liked_titles = {n.get("title", "").lower() for n in st.session_state.liked}
    disliked_titles = {n.get("title", "").lower() for n in st.session_state.disliked}
    for match in st.session_state.nonprofit_matches:
        title = match.get("title", "").lower()
        if title not in liked_titles and title not in disliked_titles:
            filtered_matches.append(match)
    st.session_state.nonprofit_matches = filtered_matches

# Use the recommendation algorithm to update the ordering of matches.
update_recommendations()







##########################################
# Swipe Interface with Callback Functions
##########################################
def dislike_candidate():
    current = st.session_state.nonprofit_matches[0]
    st.session_state.disliked.append(current)
    st.session_state.nonprofit_matches.pop(0)
    update_recommendations()
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.write("Please refresh the page manually.")

def like_candidate():
    current = st.session_state.nonprofit_matches[0]
    st.session_state.liked.append(current)
    st.session_state.nonprofit_matches.pop(0)
    update_recommendations()
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.write("Please refresh the page manually.")

if st.session_state.nonprofit_matches:
    current = st.session_state.nonprofit_matches[0]
    
    st.markdown(f"## {current.get('title', 'No Title')}")
    st.write(current.get("summary", "No summary available."))
    st.write(f"[Visit Website]({current.get('link', '#')})")
    
    col1, col2 = st.columns(2)
    col1.button("❌ Dislike", on_click=dislike_candidate)
    col2.button("❤️ Like", on_click=like_candidate)
else:
    st.write("🎉 No more nonprofit matches available!")

##########################################
# Display Swipe History
##########################################
st.write("## Your Preferences")
st.write("### ❤️ Liked")
for item in st.session_state.liked:
    st.write(f"- {item.get('title', 'No Title')}")
st.write("### ❌ Disliked")
for item in st.session_state.disliked:
    st.write(f"- {item.get('title', 'No Title')}")