import sqlite3
import math
import re
from collections import Counter
import numpy as np
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "textbooks"
SQLITE_DB_PATH = "parents.db"

# Embedding model (must match ingest.py)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Chroma safely
client = PersistentClient(path=CHROMA_DB_PATH)
try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception:
    collection = client.get_or_create_collection(COLLECTION_NAME)

# Helper function for tokenization
def tokenize(text: str):
    return re.findall(r'\w+', text.lower())

# Custom BM25 implementation for zero-dependency lexical search
class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size > 0 else 0
        self.doc_freqs = []
        self.doc_len = []
        self.nd = {}  # word -> number of docs containing word
        
        for doc in corpus:
            self.doc_len.append(len(doc))
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                self.nd[word] = self.nd.get(word, 0) + 1
                
        self.idf = {}
        for word, freq in self.nd.items():
            # Standard BM25 IDF formula
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query):
        scores = []
        for i in range(self.corpus_size):
            score = 0.0
            doc_freq = self.doc_freqs[i]
            d_len = self.doc_len[i]
            for word in query:
                if word in doc_freq:
                    tf = doc_freq[word]
                    # BM25 scoring formula
                    num = self.idf.get(word, 0) * tf * (self.k1 + 1)
                    den = tf + self.k1 * (1 - self.b + self.b * (d_len / self.avgdl))
                    score += num / den
            scores.append(score)
        return scores

# Caching for child chunks and BM25 index
_bm25_instance = None
_all_child_chunks = []
_id_to_chunk = {}

def load_bm25_and_chunks():
    global _bm25_instance, _all_child_chunks, _id_to_chunk
    if _bm25_instance is not None:
        return _bm25_instance, _all_child_chunks, _id_to_chunk
    
    print("Loading child chunks from Chroma to build BM25 index...")
    
    # Fetch in batches to prevent "too many SQL variables" sqlite error in Chroma
    ids = []
    documents = []
    metadatas = []
    embeddings = []
    
    limit = 2000
    offset = 0
    total_count = collection.count()
    print(f"Total chunks to load: {total_count}")
    
    while True:
        batch_results = collection.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas", "embeddings"]
        )
        
        batch_ids = batch_results.get("ids", [])
        if not batch_ids:
            break
            
        ids.extend(batch_ids)
        documents.extend(batch_results.get("documents", []))
        metadatas.extend(batch_results.get("metadatas", []))
        
        batch_embs = batch_results.get("embeddings", None)
        if batch_embs is not None and len(batch_embs) > 0:
            embeddings.extend(batch_embs)
            
        offset += len(batch_ids)
        if len(batch_ids) < limit:
            break
            
    _all_child_chunks = []
    _id_to_chunk = {}
    tokenized_corpus = []
    
    for i in range(len(ids)):
        chunk_doc = {
            "id": ids[i],
            "text": documents[i],
            "metadata": metadatas[i],
            "embedding": embeddings[i] if embeddings is not None and i < len(embeddings) else None
        }
        _all_child_chunks.append(chunk_doc)
        _id_to_chunk[ids[i]] = chunk_doc
        tokenized_corpus.append(tokenize(documents[i]))
        
    print(f"Loaded {len(_all_child_chunks)} child chunks. Building BM25 index...")
    _bm25_instance = BM25(tokenized_corpus)
    print("BM25 index built successfully.")
    
    return _bm25_instance, _all_child_chunks, _id_to_chunk

def retrieve(query: str, top_k: int = 3):
    """
    Retrieve the most relevant parent chunks from SQLite database
    using Hybrid retrieval (dense similarity + sparse lexical search) over child chunks.

    Returns:
        List of dictionaries containing parent text, source, page, similarity score, and rrf_score.
    """
    bm25, child_chunks, id_to_chunk = load_bm25_and_chunks()
    if not child_chunks:
        return []

    # 1. Dense Semantic Search
    query_embedding = embedding_model.encode(query).tolist()
    
    # Query Chroma for top dense matches
    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20,
        include=["documents", "metadatas", "distances"]
    )
    
    dense_ids = dense_results["ids"][0] if dense_results["ids"] else []
    dense_distances = dense_results["distances"][0] if dense_results["distances"] else []
    
    dense_ranks = {}
    dense_scores = {}
    for rank, (doc_id, dist) in enumerate(zip(dense_ids, dense_distances)):
        dense_ranks[doc_id] = rank
        dense_scores[doc_id] = round(1 - dist, 4)

    # 2. Sparse Lexical Search (BM25)
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    
    # Pair scores with indexes, filtering out non-matching chunks
    scored_chunks = [(score, idx) for idx, score in enumerate(bm25_scores) if score > 0.0]
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    sparse_ranks = {}
    for rank, (score, idx) in enumerate(scored_chunks[:20]):
        doc_id = child_chunks[idx]["id"]
        sparse_ranks[doc_id] = rank

    # 3. Reciprocal Rank Fusion (RRF)
    # Combine the ranks of dense and sparse search
    candidate_ids = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))
    rrf_scores = []
    
    for doc_id in candidate_ids:
        d_rank = dense_ranks.get(doc_id, None)
        s_rank = sparse_ranks.get(doc_id, None)
        
        score_dense = 1.0 / (d_rank + 60.0) if d_rank is not None else 0.0
        score_sparse = 1.0 / (s_rank + 60.0) if s_rank is not None else 0.0
        
        rrf_score = score_dense + score_sparse
        rrf_scores.append((rrf_score, doc_id))
        
    # Sort candidates by RRF score descending
    rrf_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Retrieve top candidates for parent lookup (take 15 to allow for parent deduplication)
    top_candidates = rrf_scores[:15]

    # 4. Fetch and Deduplicate Parent Chunks
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    retrieved_parents = {}
    q_emb = np.array(query_embedding)
    
    for rrf_score, doc_id in top_candidates:
        chunk = id_to_chunk[doc_id]
        parent_id = chunk["metadata"]["parent_id"]
        
        # Calculate/retrieve semantic similarity score for the threshold check in chat.py
        if doc_id in dense_scores:
            sem_score = dense_scores[doc_id]
        else:
            # For BM25-only matches, calculate cosine similarity manually using cached embeddings
            if chunk["embedding"] is not None:
                doc_emb = np.array(chunk["embedding"])
                dot = np.dot(q_emb, doc_emb)
                norm_q = np.linalg.norm(q_emb)
                norm_d = np.linalg.norm(doc_emb)
                sem_score = float(dot / (norm_q * norm_d)) if norm_q > 0 and norm_d > 0 else 0.0
                sem_score = round(sem_score, 4)
            else:
                sem_score = 0.0

        if parent_id in retrieved_parents:
            # If parent already fetched, update to highest semantic score and RRF score
            if sem_score > retrieved_parents[parent_id]["score"]:
                retrieved_parents[parent_id]["score"] = sem_score
            if rrf_score > retrieved_parents[parent_id]["rrf_score"]:
                retrieved_parents[parent_id]["rrf_score"] = rrf_score
            continue

        cursor.execute("SELECT text, source, page FROM parents WHERE id = ?", (parent_id,))
        row = cursor.fetchone()
        if row:
            parent_text, source, page = row
            retrieved_parents[parent_id] = {
                "text": parent_text,
                "source": source,
                "page": page,
                "score": sem_score,
                "rrf_score": rrf_score
            }

    conn.close()
    
    # Sort unique parents by RRF score descending
    final_results = list(retrieved_parents.values())
    final_results.sort(key=lambda x: x["rrf_score"], reverse=True)
    
    return final_results[:top_k]

if __name__ == "__main__":
    question = input("Ask a question: ")
    results = retrieve(question)

    print("\nTop Results\n")
    for i, item in enumerate(results, start=1):
        print("=" * 80)
        print(f"Result {i}")
        print(f"RRF Score: {round(item['rrf_score'], 5)}")
        print(f"Sem Score: {item['score']}")
        print(f"Source   : {item['source']}")
        print(f"Page     : {item['page']}")
        print("-" * 80)
        print(item["text"])
        print()
