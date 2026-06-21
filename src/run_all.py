"""
Complete RAG Pipeline with Stratified Sampling
"""

import pandas as pd
import numpy as np
import re
import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
from sampling import StratifiedSampler, create_stratified_sample

def simple_chunk(text, chunk_size=500, overlap=50):
    """Simple chunking by sentences"""
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if overlap > 0 and current_chunk:
                words = current_chunk.split()
                overlap_text = " ".join(words[-5:]) if len(words) > 5 else current_chunk
                current_chunk = overlap_text + " " + sentence + " "
            else:
                current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def main():
    print("="*60)
    print("COMPLETE RAG PIPELINE WITH STRATIFIED SAMPLING")
    print("="*60)
    
    # 1. Create stratified sample (10,000-15,000 complaints)
    print("\n[1/6] Creating stratified sample...")
    
    # Check if sample already exists
    if os.path.exists('data/sampled_complaints.csv'):
        print("Loading existing sample...")
        df_sampled = pd.read_csv('data/sampled_complaints.csv')
        print(f"Loaded {len(df_sampled)} records from existing sample")
    else:
        print("Creating new stratified sample...")
        df_sampled = create_stratified_sample(
            data_path='data/raw/complaints.csv',
            sample_size=15000,
            output_path='data/sampled_complaints.csv',
            random_state=42
        )
    
    # 2. Find narrative column
    narrative_col = None
    for col in df_sampled.columns:
        if 'narrative' in col.lower():
            narrative_col = col
            break
    if narrative_col is None:
        narrative_col = 'Issue'
    print(f"\nUsing narrative column: '{narrative_col}'")
    
    # 3. Clean text
    print("\n[2/6] Cleaning text...")
    
    def clean_text(x):
        if pd.isna(x) or not isinstance(x, str):
            return ""
        x = x.lower()
        x = re.sub(r'[^a-zA-Z0-9\s.]', ' ', x)
        x = re.sub(r'\s+', ' ', x).strip()
        return x
    
    df_sampled['cleaned_text'] = df_sampled[narrative_col].apply(clean_text)
    df_sampled = df_sampled[df_sampled['cleaned_text'].str.len() > 10].copy()
    print(f"After cleaning: {len(df_sampled)} complaints")
    
    # Save filtered data
    df_sampled.to_csv('data/filtered_complaints.csv', index=False)
    print("Saved filtered_complaints.csv")
    
    # 4. Chunk
    print("\n[3/6] Chunking...")
    all_chunks = []
    
    for idx, row in tqdm(df_sampled.iterrows(), total=len(df_sampled)):
        text = row['cleaned_text']
        if len(text) < 20:
            continue
        chunks = simple_chunk(text, chunk_size=500, overlap=50)
        if not chunks:
            chunks = [text]
        for c_idx, chunk in enumerate(chunks):
            all_chunks.append({
                'complaint_id': row.get('Complaint ID', idx),
                'chunk_index': c_idx,
                'total_chunks': len(chunks),
                'text': chunk.strip(),
                'product_category': row.get('Product_Category', 'Unknown'),
                'issue': row.get('Issue', ''),
                'company': row.get('Company', '')
            })
    
    chunked_df = pd.DataFrame(all_chunks)
    chunked_df.to_csv('data/chunked_complaints.csv', index=False)
    print(f"Created {len(all_chunks)} chunks")
    print("Saved chunked_complaints.csv")
    
    if len(all_chunks) == 0:
        print("\n❌ No chunks created!")
        return
    
    # 5. Generate embeddings
    print("\n[4/6] Generating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    dim = model.get_embedding_dimension()
    print(f"Model dimension: {dim}")
    
    texts = [c['text'] for c in all_chunks]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    print(f"Generated embeddings: {embeddings.shape}")
    
    # 6. Build FAISS index
    print("\n[5/6] Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    embeddings_float = embeddings.astype('float32')
    index.add(embeddings_float)
    
    os.makedirs('vector_store/faiss', exist_ok=True)
    faiss.write_index(index, 'vector_store/faiss/index.faiss')
    
    # Save metadata
    with open('vector_store/faiss/metadata.pkl', 'wb') as f:
        pickle.dump(all_chunks, f)
    
    print(f"\n✅ DONE! Vector store saved with {len(all_chunks)} chunks")
    
    # 7. Test search
    print("\n[6/6] Testing search...")
    test_queries = [
        "Why are customers unhappy with credit card fees?",
        "What are the main issues with personal loans?",
        "Tell me about savings account complaints"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        q_emb = model.encode([query], normalize_embeddings=True)
        scores, indices = index.search(q_emb.astype('float32'), min(3, len(all_chunks)))
        
        if len(indices) > 0 and len(indices[0]) > 0:
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(all_chunks):
                    print(f"  Result {i+1} (Similarity: {score:.3f}):")
                    print(f"    Product: {all_chunks[idx]['product_category']}")
                    print(f"    Text: {all_chunks[idx]['text'][:100]}...")
    
    print("\n" + "="*60)
    print("✅ SUCCESS! Vector store is ready.")
    print(f"Total chunks: {len(all_chunks)}")
    print("Files saved:")
    print("  - data/sampled_complaints.csv (stratified sample)")
    print("  - data/filtered_complaints.csv (cleaned)")
    print("  - data/chunked_complaints.csv (chunks)")
    print("  - vector_store/faiss/index.faiss (FAISS index)")
    print("  - vector_store/faiss/metadata.pkl (metadata)")

if __name__ == "__main__":
    main()