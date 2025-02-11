import numpy as np 
import pandas as pd 
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Verify that the keys are loaded
if not GROQ_API_KEY:
    raise ValueError("API keys not found! Check your .env file.")


# converting each row into a text chunks
df = pd.read_csv("Hubaix.csv")
texts = df.astype(str).apply(lambda row: " | ".join(row), axis = 1).tolist()


model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, convert_to_numpy=True)


dimension = len(embeddings[0])  # Get embedding size
index = faiss.IndexFlatL2(dimension)

# Add embeddings
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "faiss_hubaix_index.bin")



def retrieve_similar_rows(query, k=5):
    query_embedding = model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_embedding, k)
    return [texts[i] for i in indices[0]]

# query = "How can a customer apply for a loan?"
# retrieved_data = retrieve_similar_rows(query)

# print("Retrieved Rows:\n", retrieved_data)


GROQ_MODEL = "llama3-8b-8192"  

def generate_response(query):
    context = "\n".join(retrieve_similar_rows(query))

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful banking assistant."},
            {"role": "user", "content": f"Using this context:\n{context}\nAnswer: {query}"}
        ]
    }

    # 🔹 Use the correct Groq API endpoint (update if needed)
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)

    # 🔍 Print response for debugging
    print("API Response:", response.status_code, response.text)

    try:
        return response.json()["choices"][0]["message"]["content"].strip()
    except KeyError:
        return f"Error: Unexpected response format {response.json()}"


# while True:
#     query = input("\nYou: ")
#     if query.lower() in ["exit", "quit"]:
#         print("Chatbot: Goodbye! 👋")
#         break
#     response = generate_response(query)
#     print("\n", response)





app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/chatbot")
def chatbot(request: QueryRequest):
    response = generate_response(request.query)
    return {"response": response}

# Run with: uvicorn chatbot_api:app --reload
if __name__ == "__main__":
    uvicorn.run("embeddings:app", host="0.0.0.0", port=5000, reload=True)
