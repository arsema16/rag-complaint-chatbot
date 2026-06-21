"""
COMPLETE RAG PIPELINE - FIXED VERSION
"""

import pandas as pd
import numpy as np
import re
import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

print("="*60)
print("COMPLETE RAG PIPELINE - FIXED VERSION")
print("="*60)

# Simple chunking function
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

# 1. Load MORE data
print("\n[1/5] Loading data...")
try:
    # Load 50,000 rows instead of 5,000
    df = pd.read_csv('data/raw/complaints.csv', nrows=50000, low_memory=False)
    print(f"Loaded {len(df)} records")
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please make sure data is downloaded.")
    exit(1)

# Check available columns
print(f"\nAvailable columns: {df.columns.tolist()}")

# Find narrative column
narrative_col = None
for col in df.columns:
    if 'narrative' in col.lower():
        narrative_col = col
        break
if narrative_col is None:
    narrative_col = 'Issue'  # Fallback
print(f"Using narrative column: '{narrative_col}'")

# Check if narrative column has data
if narrative_col in df.columns:
    non_empty = df[narrative_col].notna().sum()
    print(f"Non-empty narratives: {non_empty:,} out of {len(df):,}")

# Filter to target products
def map_product(x):
    if pd.isna(x):
        return 'Other'
    x = str(x).lower()
    if 'credit card' in x:
        return 'Credit Card'
    elif 'loan' in x or 'payday' in x:
        return 'Personal Loan'
    elif 'savings' in x or 'checking' in x:
        return 'Savings Account'
    elif 'money' in x or 'transfer' in x:
        return 'Money Transfer'
    return 'Other'

df['Product_Category'] = df['Product'].apply(map_product)
df_filtered = df[df['Product_Category'] != 'Other'].copy()
print(f"\nFiltered to {len(df_filtered)} complaints")

print("\nProduct distribution:")
print(df_filtered['Product_Category'].value_counts())

# 2. Clean text - Keep ALL narratives, even short ones
print("\n[2/5] Cleaning text...")

def clean_text(x):
    if pd.isna(x) or not isinstance(x, str):
        return ""
    x = x.lower()
    # Remove special characters but keep text
    x = re.sub(r'[^a-zA-Z0-9\s.]', ' ', x)
    x = re.sub(r'\s+', ' ', x).strip()
    return x

# Apply cleaning
df_filtered['cleaned_text'] = df_filtered[narrative_col].apply(clean_text)

# Keep complaints with at least 10 characters (not 20)
df_filtered = df_filtered[df_filtered['cleaned_text'].str.len() > 10].copy()
print(f"After cleaning: {len(df_filtered)} complaints")

if len(df_filtered) == 0:
    print("\n❌ No complaints found with narratives!")
    print("Trying alternative approach...")
    
    # Try using 'Issue' column instead
    if narrative_col != 'Issue':
        print("Trying 'Issue' column instead...")
        df['cleaned_text'] = df['Issue'].apply(clean_text)
        df_filtered = df[df['Product_Category'] != 'Other'].copy()
        df_filtered = df_filtered[df_filtered['cleaned_text'].str.len() > 10].copy()
        print(f"Found {len(df_filtered)} complaints using 'Issue' column")

if len(df_filtered) == 0:
    print("\n❌ Still no complaints found!")
    print("Please check your data file.")
    print("Try downloading the data again:")
    print("  python download_data.py")
    exit(1)

# Save filtered data
os.makedirs('data', exist_ok=True)
df_filtered.to_csv('data/filtered_complaints.csv', index=False)
print("Saved filtered_complaints.csv")

# 3. Chunk
print("\n[3/5] Chunking...")
all_chunks = []

for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
    text = row['cleaned_text']
    if len(text) < 20:
        continue
    chunks = simple_chunk(text, chunk_size=500, overlap=50)
    if not chunks:
        chunks = [text]  # Use whole text if chunking fails
    for c_idx, chunk in enumerate(chunks):
        all_chunks.append({
            'complaint_id': row.get('Complaint ID', idx),
            'chunk_index': c_idx,
            'total_chunks': len(chunks),
            'text': chunk.strip(),
            'product_category': row['Product_Category'],
            'issue': row.get('Issue', ''),
            'company': row.get('Company', '')
        })

chunked_df = pd.DataFrame(all_chunks)
chunked_df.to_csv('data/chunked_complaints.csv', index=False)
print(f"Created {len(all_chunks)} chunks")
print("Saved chunked_complaints.csv")

if len(all_chunks) == 0:
    print("\n❌ No chunks created!")
    exit(1)

# 4. Generate embeddings
print("\n[4/5] Generating embeddings...")
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

# 5. Create FAISS index
print("\n[5/5] Building FAISS index...")
index = faiss.IndexFlatIP(dim)
embeddings_float = embeddings.astype('float32')
index.add(embeddings_float)

os.makedirs('vector_store/faiss', exist_ok=True)
faiss.write_index(index, 'vector_store/faiss/index.faiss')

# Save metadata
with open('vector_store/faiss/metadata.pkl', 'wb') as f:
    pickle.dump(all_chunks, f)

print(f"\n✅ DONE! Vector store saved with {len(all_chunks)} chunks")
print(f"Files saved to: vector_store/faiss/")

# Test search
print("\n" + "="*60)
print("TESTING SEARCH")
print("="*60)

test_queries = [
    "Why are customers unhappy with credit card fees?",
    "What are the main issues with personal loans?",
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
    else:
        print("  No results found")

print("\n" + "="*60)
print("✅ SUCCESS! Vector store is ready.")
print(f"Total chunks: {len(all_chunks)}")
print("You can now use it for RAG queries.")