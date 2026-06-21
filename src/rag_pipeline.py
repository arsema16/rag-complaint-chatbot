\"\"\"
RAG Pipeline - Query and Answer System
\"\"\"

import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGPipeline:
    def __init__(self, vector_store_path='vector_store/faiss'):
        \"\"\"Initialize RAG pipeline\"\"\"
        print(\"Loading vector store...\")
        
        # Load model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dim = self.model.get_embedding_dimension()
        
        # Load FAISS index
        self.index = faiss.read_index(f'{vector_store_path}/index.faiss')
        
        # Load metadata
        with open(f'{vector_store_path}/metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f\"Loaded {len(self.metadata)} chunks\")
    
    def retrieve(self, query, k=5):
        \"\"\"Retrieve top-k relevant chunks\"\"\"
        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(q_emb.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                results.append({
                    'text': self.metadata[idx]['text'],
                    'metadata': self.metadata[idx],
                    'score': float(score)
                })
        return results
    
    def generate_answer(self, query, context_chunks):
        \"\"\"Generate answer from context\"\"\"
        if context_chunks:
            return context_chunks[0]['text']
        return \"No relevant information found.\"
    
    def query(self, question, k=3):
        \"\"\"Complete RAG query\"\"\"
        chunks = self.retrieve(question, k)
        answer = self.generate_answer(question, chunks)
        
        return {
            'question': question,
            'answer': answer,
            'sources': chunks
        }

if __name__ == \"__main__\":
    rag = RAGPipeline()
    questions = [
        \"Why are customers unhappy with credit card fees?\",
        \"What are the main issues with personal loans?\"
    ]
    
    print(\"\\n\" + \"=\"*60)
    print(\"RAG PIPELINE TEST\")
    print(\"=\"*60)
    
    for q in questions:
        print(f\"\\n❓ Question: {q}\")
        result = rag.query(q, k=3)
        print(f\"\\n📝 Answer: {result['answer'][:200]}...\")
        print(f\"\\n📎 Sources: {len(result['sources'])} chunks retrieved\")
