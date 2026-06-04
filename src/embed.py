import os
import lancedb
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_or_create_embeddings(documents, db_path="db/embeddings"):
    # Loads embeddings from LanceDB if they exist, otherwise embeds documents and saves them.
    db = lancedb.connect(os.path.abspath(db_path))

    if "documents" in db.table_names():
        print("Loading embeddings from database...")
        table = db.open_table("documents")
        return table.to_pandas().to_dict(orient="records")
    
    print("Embedding documents for the first time...")
    embedded = []

    for doc in documents:
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=doc["text"]
        )
        embedded.append({
            "filename": doc["filename"],
            "text": doc["text"],
            "embedding": response.embeddings[0].values
        })

    table = db.create_table("documents", data=embedded)
    print("Embeddings saved to database.")

    return embedded 
