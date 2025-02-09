from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import openai
import asyncio
import httpx
import urllib.parse
import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import pandas as pd
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()
openai.api_base = os.getenv("OPENAI_BASE_URL")
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise Exception("Missing OpenAI API Key! Make sure it's set in your .env file.")

# Global dictionary to simulate session state
session_state = {
    "messages": [{"role": "assistant", "content": "Hello! Before we start, can you tell me a little bit about your school?"}],
    "liked": [],
    "disliked": [],
    "nonprofit_matches": [],
    "latest_summary": "",
    "summary_hash": ""
}

SERP_API_KEY = "1c92173370dc002dd7a0053eb47eeef87f9cce81bc61ccdcc766c1672a4b0087"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_csvs(directory: str) -> pd.DataFrame:
    """
    Loads all CSV files from the specified directory and combines them into a single DataFrame.
    """
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    df_list = []
    for file in csv_files:
        print(f"Loading file: {file}")
        df = pd.read_csv(file)
        df_list.append(df)
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

def generate_description(row) -> str:
    """
    Generate a description for the nonprofit.
    Prefer to use the mission field (F9_01_ACT_GVRN_ACT_MISSION) if available
    and not "NA"; otherwise, fall back to using the organization name (ORG_NAME_L1).
    """
    mission = row.get("F9_01_ACT_GVRN_ACT_MISSION", "")
    if pd.notnull(mission) and str(mission).strip() and str(mission).strip().upper() != "NA":
        return str(mission).strip()
    else:
        org_name = row.get("ORG_NAME_L1", "")
        if pd.notnull(org_name) and str(org_name).strip():
            return f"{org_name.strip()} is a nonprofit organization."
        return "No description available."
    
def generate_metadata(row) -> dict:
    """
    Generate metadata for the nonprofit from the CSV row.
    """
    return {
        "org_name": row.get("ORG_NAME_L1", ""),
        "url": row.get("URL", ""),
        "mission": row.get("F9_01_ACT_GVRN_ACT_MISSION", "")
    }

def get_embedding(text: str) -> np.ndarray:
    """
    Generate an embedding for the given text using the OpenAI API with the
    model "amazon.titan-text-embeddings.v2".
    """
    response = openai.Embedding.create(
         input=text,
         model="amazon.titan-text-embeddings.v2"
    )
    return np.array(response["data"][0]["embedding"])

# Asynchronous Functions for SERP API Calls

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
    # if not text:
        
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

def extract_keywords(text, top_n=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    top_keywords = feature_array[tfidf_sorting][:top_n]
    return " ".join(top_keywords)

def get_embedding_keywords(text):
    keywords = extract_keywords(text)
    response = openai.embeddings.create(
        input=keywords,
        model="amazon.titan-text-embeddings.v2"
    )
    return np.array(response.data[0].embedding)

def update_recommendations():
    if not session_state.get("liked") and not session_state.get("disliked"):
        return
    new_recommendations = []
    for candidate in session_state.get("nonprofit_matches", []):
        if candidate in session_state.get("liked", []) or candidate in session_state.get("disliked", []):
            continue
        candidate_embedding = get_embedding_keywords(candidate["summary"])
        liked_embeddings = ([get_embedding_keywords(n["summary"]) for n in session_state.get("liked", [])] if session_state.get("liked") else [])
        disliked_embeddings = ([get_embedding_keywords(n["summary"]) for n in session_state.get("disliked", [])] if session_state.get("disliked") else [])
        liked_similarities = ([cosine_similarity([candidate_embedding], [emb])[0][0] for emb in liked_embeddings] if liked_embeddings else [0])
        disliked_similarities = ([cosine_similarity([candidate_embedding], [emb])[0][0] for emb in disliked_embeddings] if disliked_embeddings else [0])
        avg_liked_similarity = np.mean(liked_similarities) if liked_similarities else 0
        penalty = 0
        for d_sim in disliked_similarities:
            if d_sim >= 0.5:
                penalty = 1
                break
        if penalty == 0:
            new_recommendations.append((candidate, avg_liked_similarity))
    new_recommendations.sort(key=lambda x: x[1], reverse=True)
    session_state["nonprofit_matches"] = [x[0] for x in new_recommendations]

##########################################
# API Endpoints
##########################################

# Chat Endpoint
class ChatRequest(BaseModel):
    messages: List[dict]

class ChatResponse(BaseModel):
    response: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    SYSTEM_MESSAGE = {
        "role": "system",
        "content": (
            "You are an AI assistant whose sole purpose is to match schools and students in need with nonprofit organizations that can help them. "
            "Your task is to ask only relevant questions to collect essential information about the school or student. "
            "Focus on asking for details such as the school's location, school size, and most importantly, what specific kind of help they need. "
            "If the user provides irrelevant information or asks unrelated questions, kindly prompt them to provide the necessary details."
            "Ask one question at a time to not overwhelm the user."
            "Do not ask too many questions. Once the student's need is clearly identified, try to find a match"
            "Do not directly tell the student about a non-profit even after identifying it. "
            "When you feel like you have found a few good matchs, instead just say 'Found a match'"
        )
    }
    messages = [SYSTEM_MESSAGE] + request.messages
    try:
        response = openai.chat.completions.create(
            model="anthropic.claude-3.5-haiku",
            messages=messages,
        )
        assistant_reply = response.choices[0].message.content
        session_state["messages"] = request.messages + [{"role": "assistant", "content": assistant_reply}]
        return ChatResponse(response=assistant_reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Summarize Endpoint
class SummarizeRequest(BaseModel):
    conversation: str

class SummarizeResponse(BaseModel):
    summary: str
    search_query: str

@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize_endpoint(request: SummarizeRequest):
    conversation_text = request.conversation
    SYSTEM_PROMPT = (
        "You are an AI assistant whose sole purpose is to match schools and students in need with nonprofit organizations that can help them. "
        "Your task is to ask only relevant questions to collect essential information from the school, including details such as the school's location, "
        "school size, and the specific kind of help needed. "
        "If the conversation contains any externally verified information about a high school (e.g., from online searches, such as its location or size), "
        "include that information in your summary. "
        "Now, based on the following conversation—which includes both your prompts and the user's responses—provide a concise summary that captures the essential details needed for matching. "
        "You should not ask the user to provide more details, only summarize the details they have provided."
    )
    try:
        summary_response = openai.chat.completions.create(
            model="anthropic.claude-3.5-haiku",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": conversation_text}
            ],
        )
        summary = summary_response.choices[0].message.content.strip()
        QUERY_PROMPT = (
            "Based on the following summary, generate a search query that focuses on nonprofit organizations providing aid. "
            "The search query should emphasize details such as location, specific needs, and types of assistance, "
            "but it must NOT include any information about the school or its website. "
            "Only include keywords relevant to nonprofit organizations and their services. "
            "The query should only contain the support needed and the location, NOTHING ELSE. it should be a couple words or phrases at most. make sure the last word is always nonprofit "
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
        session_state["latest_summary"] = search_query
        return SummarizeResponse(summary=summary, search_query=search_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Matches Endpoint
class MatchesRequest(BaseModel):
    search_query: str = ""

class MatchesResponse(BaseModel):
    matches: List[dict]

@app.post("/api/matches", response_model=MatchesResponse)
async def matches_endpoint(request: MatchesRequest):
    query = request.search_query or session_state.get("latest_summary", "")
    if not query:
        raise HTTPException(status_code=400, detail="No search query provided")
    try:
        matches = await full_search_summarize(query)
        liked_titles = {n.get("title", "").lower() for n in session_state.get("liked", [])}
        disliked_titles = {n.get("title", "").lower() for n in session_state.get("disliked", [])}
        new_matches = [m for m in matches if m.get("title", "").lower() not in liked_titles and m.get("title", "").lower() not in disliked_titles]
        session_state["nonprofit_matches"] = new_matches
        return MatchesResponse(matches=new_matches)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
