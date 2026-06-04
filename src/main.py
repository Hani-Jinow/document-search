import os
from dotenv import load_dotenv
from google import genai 
from extract import extract_all
from embed import get_or_create_embeddings
from search import find_best_match

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def answer_question(question, context):
    # Sends the question and retrieved context to Gemini and returns a grounded answer.
    prompt = f""" You are a helpful assistant. 
    Answer the question based only on the context provided below. 
    If the answer is not in the context, say "I don't have any information about that."

    context: {context}

    Question: {question}

    Answer: """

    response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
    return response.text

def main():
    # Orchestrates the full RAG pipeline — extract, embed, search, and answer.
    print("Loading documents...")
    docs = extract_all("data/documents")

    print("Embedding documents...")
    embedded = get_or_create_embeddings(docs)

    print("\nReady! Ask me anything about your documents.")
    print("Type 'quit' to exit\n")

    while True:
        question = input("Your question: ")

        if question.lower() == "quit": 
            break

        best_doc, score = find_best_match(question, embedded)
        answer = answer_question(question, best_doc["text"])

        print(f"\nSource: {best_doc['filename']} (confidence: {score:2f})")
        print(f"Answer: {answer}\n")

if __name__ == "__main__":
    main()