import os
import uuid
import sqlite3
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_FOLDER = "data/pdfs"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "textbooks"
SQLITE_DB_PATH = "parents.db"

PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50

# Init Chroma and SQLite
import shutil

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = None
collection = None

def reset_databases():
    global client, collection
    
    # 1. Try to delete SQLite DB
    if os.path.exists(SQLITE_DB_PATH):
        try:
            os.remove(SQLITE_DB_PATH)
            print("Deleted old parents.db SQLite file.")
        except Exception as e:
            print(f"Notice: parents.db is locked/in-use, clearing records via SQL: {e}")
            conn = sqlite3.connect(SQLITE_DB_PATH)
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS parents")
            conn.commit()
            conn.close()

    # 2. Try to delete Chroma directory if we can (to prevent corruption)
    chroma_removed = False
    if os.path.exists(CHROMA_DB_PATH):
        try:
            shutil.rmtree(CHROMA_DB_PATH)
            print("Deleted old chroma_db directory to prevent corruption.")
            chroma_removed = True
        except Exception as e:
            print(f"Notice: chroma_db directory is in-use, will clear collection documents instead: {e}")

    # Initialize Chroma client
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    if chroma_removed:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
    else:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        try:
            count = collection.count()
            if count > 0:
                print(f"Clearing {count} existing chunks from Chroma collection...")
                all_ids = collection.get()["ids"]
                for i in range(0, len(all_ids), 100):
                    batch = all_ids[i:i+100]
                    collection.delete(ids=batch)
        except Exception as e:
            print(f"Warning: Could not clear Chroma collection: {e}")

reset_databases()

# Init SQLite DB for parent chunks
def init_sqlite():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    # Drop table if exists to ensure clean ingestion (in case file wasn't deleted)
    cursor.execute("DROP TABLE IF EXISTS parents")
    cursor.execute("""
        CREATE TABLE parents (
            id TEXT PRIMARY KEY,
            text TEXT,
            source TEXT,
            page INTEGER
        )
    """)
    conn.commit()
    conn.close()

init_sqlite()

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_CHUNK_SIZE,
    chunk_overlap=PARENT_CHUNK_OVERLAP
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHUNK_SIZE,
    chunk_overlap=CHILD_CHUNK_OVERLAP
)

def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    filename = Path(pdf_path).name

    print(f"\nProcessing: {filename}")

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    total_parents = 0
    total_children = 0

    for page in pages:
        page_number = page.metadata["page"] + 1
        
        # Split page content into parent chunks
        parent_chunks = parent_splitter.split_text(page.page_content)
        
        for parent_text in parent_chunks:
            parent_id = str(uuid.uuid4())
            
            # Store parent chunk in SQLite
            cursor.execute(
                "INSERT INTO parents (id, text, source, page) VALUES (?, ?, ?, ?)",
                (parent_id, parent_text, filename, page_number)
            )
            total_parents += 1
            
            # Split parent chunk into child chunks
            child_chunks = child_splitter.split_text(parent_text)
            
            for child_text in child_chunks:
                child_id = str(uuid.uuid4())
                
                # Store child chunk in Chroma
                collection.add(
                    ids=[child_id],
                    documents=[child_text],
                    metadatas=[
                        {
                            "source": filename,
                            "page": page_number,
                            "parent_id": parent_id
                        }
                    ]
                )
                total_children += 1

    conn.commit()
    conn.close()
    print(f"Stored {total_parents} parents and {total_children} children")

def main():
    pdfs = list(Path(PDF_FOLDER).glob("*.pdf"))

    if not pdfs:
        print("No PDFs found.")
        return

    for pdf in pdfs:
        process_pdf(str(pdf))

    print("\nDone!")
    print(f"Total Child Chunks Stored in Chroma: {collection.count()}")

if __name__ == "__main__":
    main()