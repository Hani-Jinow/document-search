# Document Search with RAG

A lightweight RAG (Retrieval-Augmented Generation) pipeline built in Python. Ask questions in plain English and get answers grounded in your own documents — no hallucinations, no guessing.

Built as a learning project to understand how AI search actually works under the hood.

---

## What it does

Most AI tools answer questions from their training data. This pipeline answers questions from *your* documents only.

You provide a folder of `.txt` files. The pipeline:
1. Reads and extracts the text from each file
2. Converts each document into a vector embedding — a list of 3072 numbers representing its meaning
3. Takes your question, converts it to an embedding the same way
4. Finds the document whose meaning is mathematically closest to your question (cosine similarity)
5. Sends that document + your question to Gemini, which returns a grounded answer

If the answer isn't in your documents, it says so — rather than making something up.

---

## How RAG works

```
Your documents
      ↓
Extract text → Convert to embeddings → Store in memory
                                              ↓
Your question → Convert to embedding → Find closest match → Send to Gemini → Answer
```

The key insight: instead of searching for exact keywords, the pipeline searches for *meaning*. Asking "is banana a fruit?" finds a document about berries — because the concepts are semantically related, even if the words don't match.

---

## Project structure

```
document-search/
│
├── src/
│   ├── extract.py        # Reads .txt files from a folder
│   ├── embed.py          # Converts text to vector embeddings via Gemini API
│   ├── search.py         # Cosine similarity search across embedded documents
│   └── main.py           # Orchestrates the pipeline and handles user input
│
├── data/
│   └── documents/        # Add your .txt files here
│
├── .env                  # API key (not committed to GitHub)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/document-search.git
cd document-search
```

### 2. Install dependencies

```bash
pip install google-genai python-dotenv numpy
```

### 3. Add your Gemini API key

Create a `.env` file in the root folder:

```
GEMINI_API_KEY=your_key_here
```

Get a free API key at [aistudio.google.com](https://aistudio.google.com)

### 4. Add your documents

Drop any `.txt` files into `data/documents/`. The pipeline will automatically pick them up — no code changes needed.

### 5. Run it

```bash
python src/main.py
```

Then ask questions about your documents.

---

## Example

Documents in `data/documents/`:
- `octopus.txt` — facts about octopuses
- `berries.txt` — facts about berries
- `chess.txt` — facts about chess
- `brain.txt` — facts about the human brain

```
Your question: how many hearts does an octopus have?
Source: octopus.txt (confidence: 0.86)
Answer: Octopuses have three hearts.

Your question: what fruit is technically a berry?
Source: berries.txt (confidence: 0.79)
Answer: Bananas are berries in botanical terms.
```

---

## Tech stack

| Tool | Purpose |
|---|---|
| Python | Core pipeline logic |
| Google Gemini API | Embeddings + answer generation |
| `gemini-embedding-2` | Converts text to 3072-dimensional vectors |
| `gemini-2.5-flash` | Generates answers from retrieved context |
| NumPy | Cosine similarity calculation |
| python-dotenv | Secure API key management |

---

## What I learned

- How vector embeddings represent meaning as numbers
- Why semantic search finds relevant results even when exact words don't match
- How cosine similarity measures the mathematical distance between two vectors
- How RAG grounds AI answers in specific documents rather than training data
- How to structure a multi-file Python project with separation of concerns
- Why encoding (UTF-8), API deprecation, and rate limits are real engineering problems

---

## What's next

- [ ] Add PDF support using `pdfplumber`
- [ ] Persist embeddings to disk so documents don't re-embed on every run
- [ ] Add a simple web interface with FastAPI
- [ ] Store embeddings in a proper vector database (ChromaDB or Pinecone)
