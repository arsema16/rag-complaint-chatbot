"""
FAISS Vector Store - LIGHTNING FAST
"""

import pandas as pd
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

def main():
    print("="*60)
    print("FAISS VECTOR STORE (FAST)")
    print("="*60)
    
    # Load chunks
    print("\nLoading chunks...")
    try:
        chunked_df = pd.read_csv('data/chunked_complaints.csv')
        print(f"Loaded {len(chunked_df)} chunks")
    except:
        print("No chunked data found. Creating from raw...")
        df = pd.read_csv('data/raw/complaints.csv', nrows=3000, low_memory=False)
        narrative_col = 'Consumer complaint narrative' if 'Consumer complaint narrative' in df.columns else 'Issue'
        df['text'] = df[narrative_col].fillna('').astype(str)
        df = df[df['text'].str.len() > 10]
        
        # Simple chunking
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        chunker = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = []
        for idx, row in df.iterrows():
            text = str(row['text'])
            if len(text) < 50:
                continue
            for c_idx, chunk in enumerate(chunker.split_text(text)):
                chunks.append({
                    'complaint_id': row.get('Complaint ID', idx),
                    'chunk_index': c_idx,
                    'total_chunks': len(chunker.split_text(text)),
                    'text': chunk.strip(),
                    'product_category': 'Credit Card' if 'credit card' in str(row.get('Product', '')).lower() else 'Other',
                    'issue': row.get('Issue', ''),
                })
        chunked_df = pd.DataFrame(chunks)
        chunked_df.to_csv('data/chunked_complaints.csv', index=False)
        print(f"Created {len(chunked_df)} chunks")
    
    # Load embedding model (already installed)
    print("\nLoading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Dimension: {dim}")
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    texts = chunked_df['text'].tolist()
    embeddings = model.encode(
        texts,
        batch_size=64,  # Faster
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    print(f"Generated embeddings: {embeddings.shape}")
    
    # Create FAISS index
    print("\nCreating FAISS index...")
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    index.add(embeddings.astype('float32'))
    print(f"Added {index.ntotal} vectors to index")
    
    # Save FAISS index and metadata
    print("\nSaving...")
    os.makedirs('vector_store/faiss', exist_ok=True)
    faiss.write_index(index, 'vector_store/faiss/index.faiss')
    
    # Save metadata separately
    metadata = chunked_df[['complaint_id', 'product_category', 'issue', 'text']].to_dict('records')
    with open('vector_store/faiss/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("Saved to vector_store/faiss/")
    
    # Test search
    print("\n" + "="*60)
    print("TESTING SEARCH")
    print("="*60)
    query = "Why are customers unhappy with credit card fees?"
    query_embedding = model.encode([query], normalize_embeddings=True)
    
    scores, indices = index.search(query_embedding.astype('float32'), 3)
    
    print(f"\nQuery: {query}")
    print("\nTop results:")
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < len(metadata):
            print(f"\nResult {i+1} (Similarity: {score:.3f})")
            print(f"Product: {metadata[idx]['product_category']}")
            print(f"Text: {metadata[idx]['text'][:150]}...")
    
    print("\n" + "="*60)
    print("DONE! Vector store ready.")
    print(f"Total chunks: {len(metadata)}")

if __name__ == "__main__":
    main()