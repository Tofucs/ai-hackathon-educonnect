from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
import os
import uvicorn
import openai
import chromadb
from chromadb.config import Settings
import numpy as np
import pandas as pd
import glob
import sys

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

collection = None

# Pydantic models for search requests and results.
class SearchRequest(BaseModel):
    query: str
    k: int = 5  # Number of results to return (default is 5)

class NonprofitResult(BaseModel):
    org_name: str
    url: str
    mission: str
    score: float

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

# ONLY FOR F9-P01 FILE FROM NCCS
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

def get_embedding(text: str) -> List[float]:
    """
    Generate an embedding for the given text using the OpenAI API with the
    model "amazon.titan-text-embeddings.v2".
    """
    response = openai.Embedding.create(
         input=text,
         model="amazon.titan-text-embeddings.v2"
    )
    return np.array(response.data[0].embedding)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection
    try:
        # Creates Chroma client with persistent storage.
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",  # Persistent backend using DuckDB + Parquet
            persist_directory="./chroma_db"     # Directory where the vector store will be persisted
        ))
        try:
            collection = client.get_collection(name="nonprofits")
            print("Loaded existing Chroma collection.")
        except Exception:
            collection = client.create_collection(name="nonprofits")
            print("Created new Chroma collection.")

            data_directory = os.path.join(os.path.dirname(__file__), 'data')
            df = load_csvs(data_directory)
            if df.empty:
                print("No CSV files found in the data directory.")
            else:
                # find possible descriptions if in data
                df["combined_text"] = df.apply(generate_description, axis=1)
                # same for metadata
                metadata_list = df.apply(generate_metadata, axis=1).tolist()
                documents = df["combined_text"].tolist()
                ids = [str(i) for i in range(len(documents))]
                # Generate embeddings for each document.
                embeddings = [get_embedding(doc) for doc in documents]
                # Add all documents, IDs, metadata, and embeddings to the Chroma collection.
                collection.add(
                    documents=documents,
                    ids=ids,
                    metadatas=metadata_list,
                    embeddings=embeddings
                )
                print("Added documents to Chroma collection.")
    except Exception as e:
        print("Error during Chroma collection setup:", e)
        raise e
    yield

# Create the FastAPI app with the custom lifespan.
app = FastAPI(lifespan=lifespan)

# Post search endpoint
@app.post("/search", response_model=List[NonprofitResult])
async def search_nonprofits(request: SearchRequest):
    print("Inside /search endpoint", file=sys.stderr)
    return [{
        "org_name": "Test Org",
        "url": "http://example.com",
        "mission": "This is a test mission.",
        "score": 0.123
    }]

# Health check endpoint.
@app.get("/health")
async def health():
    return {"status": "healthy"}


    

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    