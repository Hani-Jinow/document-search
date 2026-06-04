import os
import numpy as np
from dotenv import load_dotenv
from google import genai
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_query_embedding(query):
    # Converts a query string into a vector embedding using Gemini.
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=query
    )
    return response.embeddings[0].values

def find_best_match(query, embedded_docs):
    # Finds the most semantically similar document to the query using cosine similarity.
    query_embedding = get_query_embedding(query)
    query_vector = np.array(query_embedding)

    best_score = -1
    best_doc = None

    for doc in embedded_docs:
        doc_vector = np.array(list(doc["embedding"]))

        score = np.dot(query_vector, doc_vector) / (
            np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
        )

        if score > best_score:
            best_score = score
            best_doc = doc

    return best_doc, best_score

if __name__ == "__main__":
    from extract import extract_all
    from embed import embed_documents

    docs = extract_all("data/documents")
    embedded = embed_documents(docs)

    query = "How strong is the brain?"
    best_doc, score = find_best_match(query, embedded)

    print(f"Query: {query}")
    print(f"Best match: {best_doc['filename']}")
    print(f"Score: {score:.4f}")
    print(f"Content: {best_doc['text']}")