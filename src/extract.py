import os

# Reads all .txt files in a folder and returns a list of dicts with filename and text.
def extract_all(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append({
                    "filename": filename,
                    "text": text
                })

    return documents
