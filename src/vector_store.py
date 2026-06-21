"""
Vector store module using ChromaDB
"""

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Loaded model: {model_name}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def generate(self, texts: List[str], batch_size=32) -> np.ndarray:
        """Generate embeddings for texts"""
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        print(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

class ChromaVectorStore:
    """ChromaDB vector store"""
    
    def __init__(self, persist_dir='vector_store/chromadb', collection_name='complaints'):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {collection_name}")
    
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Add chunks to vector store"""
        print(f"Adding {len(chunks)} chunks to ChromaDB...")
        batch_size = 1000
        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            ids = [f"chunk_{j}" for j in range(i, i+len(batch))]
            documents = [chunk['text'] for chunk in batch]
            metadatas = [
                {
                    'complaint_id': chunk['complaint_id'],
                    'product_category': chunk['product_category'],
                    'chunk_index': chunk['chunk_index'],
                    'total_chunks': chunk['total_chunks'],
                    'issue': chunk.get('issue', ''),
                    'company': chunk.get('company', '')
                }
                for chunk in batch
            ]
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=batch_embeddings,
                metadatas=metadatas
            )
        print(f"Added {len(chunks)} chunks successfully")
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        formatted_results = []
        if results['documents']:
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                formatted_results.append({
                    'text': doc,
                    'metadata': metadata,
                    'similarity': 1 - distance
                })
        return formatted_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            'total_chunks': self.collection.count(),
            'collection_name': self.collection_name,
            'persist_directory': self.persist_dir
        }

if __name__ == "__main__":
    print("Loading chunked data...")
    chunked_df = pd.read_csv('data/chunked_complaints.csv')
    print(f"Loaded {len(chunked_df)} chunks")
    
    # Initialize embedding model
    embedder = EmbeddingGenerator('all-MiniLM-L6-v2')
    
    # Extract texts and chunks
    texts = chunked_df['text'].tolist()
    chunks = chunked_df.to_dict('records')
    
    # Generate embeddings
    embeddings = embedder.generate(texts)
    
    # Create vector store
    store = ChromaVectorStore(persist_dir='vector_store/chromadb')
    store.add_chunks(chunks, embeddings.tolist())
    
    # Test search
    print("\nTesting search...")
    test_question = "Why are customers unhappy with credit card fees?"
    query_embedding = embedder.generate([test_question])[0]
    
    results = store.search(query_embedding, k=3)
    
    print(f"\nQuery: {test_question}")
    print("\nTop Results:")
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (Similarity: {result['similarity']:.3f})")
        print(f"Product: {result['metadata']['product_category']}")
        print(f"Text: {result['text'][:150]}...")