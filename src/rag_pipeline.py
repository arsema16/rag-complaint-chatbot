"""
RAG Pipeline - Retrieval Augmented Generation for Complaint Analysis

This module implements the complete RAG pipeline including:
- Retriever: FAISS similarity search
- Generator: Context-based answer generation with LLM integration
- Evaluation: Quality scoring and testing
"""

import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import json
import re

class RAGPipeline:
    """
    Complete RAG Pipeline for Complaint Analysis
    
    Features:
    - Semantic search using FAISS
    - Context-based answer generation
    - Source attribution and metadata tracking
    - Qualitative evaluation capabilities
    """
    
    def __init__(self, 
                 vector_store_path: str = 'vector_store/faiss',
                 model_name: str = 'all-MiniLM-L6-v2',
                 k: int = 5):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store_path: Path to FAISS vector store
            model_name: Name of the embedding model
            k: Default number of chunks to retrieve
        """
        self.vector_store_path = vector_store_path
        self.model_name = model_name
        self.k = k
        
        # Load embedding model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()
        
        # Load vector store
        print(f"Loading vector store from: {vector_store_path}")
        self.index = faiss.read_index(f'{vector_store_path}/index.faiss')
        
        # Load metadata
        with open(f'{vector_store_path}/metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loaded {len(self.metadata)} chunks, dimension: {self.dim}")
        print(f"Default k = {self.k}")
    
    def retrieve(self, 
                 query: str, 
                 k: Optional[int] = None,
                 filter_product: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant chunks for a query.
        
        Args:
            query: User question
            k: Number of chunks to retrieve (defaults to self.k)
            filter_product: Optional product category filter
            
        Returns:
            List of retrieved chunks with metadata and scores
        """
        k = k or self.k
        
        # Generate query embedding
        query_embedding = self.model.encode(
            [query], 
            normalize_embeddings=True
        )[0]
        
        # Search
        scores, indices = self.index.search(
            query_embedding.astype('float32').reshape(1, -1), 
            k * 2  # Retrieve extra for filtering
        )
        
        # Get results with optional filtering
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                chunk = self.metadata[idx]
                
                # Apply product filter if specified
                if filter_product and chunk['product_category'] != filter_product:
                    continue
                    
                results.append({
                    'text': chunk['text'],
                    'metadata': {
                        'complaint_id': chunk['complaint_id'],
                        'product_category': chunk['product_category'],
                        'chunk_index': chunk['chunk_index'],
                        'total_chunks': chunk['total_chunks'],
                        'issue': chunk.get('issue', ''),
                        'company': chunk.get('company', '')
                    },
                    'score': float(score)
                })
                
                if len(results) >= k:
                    break
        
        return results
    
    def generate_answer(self, 
                        query: str, 
                        context_chunks: List[Dict[str, Any]],
                        use_llm: bool = False) -> str:
        """
        Generate an answer from retrieved context.
        
        Args:
            query: Original user question
            context_chunks: Retrieved chunks
            use_llm: Whether to use LLM (requires API key)
            
        Returns:
            Generated answer string
        """
        if not context_chunks:
            return "No relevant information found in the complaint database."
        
        # Build context
        context = "\n\n".join([
            f"[Source {i+1} - Product: {chunk['metadata']['product_category']}]\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # If LLM is available, use it
        if use_llm:
            return self._generate_with_llm(query, context, context_chunks)
        
        # Otherwise, use a simple template-based approach
        return self._generate_with_template(query, context, context_chunks)
    
    def _generate_with_template(self, 
                                query: str, 
                                context: str,
                                context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate answer using a template-based approach.
        
        This is a fallback when LLM is not available.
        """
        # Find the most relevant chunk
        if context_chunks:
            best_chunk = context_chunks[0]
            answer = best_chunk['text']
            
            # Add product context
            product = best_chunk['metadata']['product_category']
            issue = best_chunk['metadata'].get('issue', '')
            
            if issue:
                answer = f"For {product} complaints related to '{issue}':\n{answer}"
            else:
                answer = f"For {product} complaints:\n{answer}"
            
            # Add summary
            if len(context_chunks) > 1:
                answer += f"\n\nBased on {len(context_chunks)} relevant complaint sources."
            
            return answer
        
        return "No relevant information found."
    
    def _generate_with_llm(self, 
                           query: str, 
                           context: str,
                           context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate answer using an LLM (requires API key).
        
        This is a placeholder - implement with your preferred LLM.
        """
        # TODO: Implement LLM integration
        # Examples:
        # - OpenAI: openai.ChatCompletion.create()
        # - Hugging Face: pipeline("text-generation", model="...")
        # - Mistral: mistralai client
        # - Local: llama.cpp or Ollama
        
        prompt = f"""
You are a financial analyst assistant for CrediTrust Financial. 
Your task is to answer questions about customer complaints.

Use ONLY the following complaint excerpts to formulate your answer.
If the context doesn't contain the answer, state that you don't have 
enough information.

Context:
{context}

Question: {query}

Answer (concise and evidence-based):
"""
        
        # Placeholder - replace with actual LLM call
        return f"LLM integration not yet implemented. \n\nBased on the retrieved context:\n{context[:500]}..."
    
    def query(self, 
              question: str, 
              k: Optional[int] = None,
              filter_product: Optional[str] = None,
              use_llm: bool = False) -> Dict[str, Any]:
        """
        Complete RAG query workflow.
        
        Args:
            question: User question
            k: Number of chunks to retrieve
            filter_product: Optional product filter
            use_llm: Whether to use LLM for generation
            
        Returns:
            Dictionary with question, answer, sources, and metadata
        """
        # 1. Retrieve relevant chunks
        chunks = self.retrieve(question, k=k, filter_product=filter_product)
        
        # 2. Generate answer
        answer = self.generate_answer(question, chunks, use_llm=use_llm)
        
        # 3. Return results
        return {
            'question': question,
            'answer': answer,
            'sources': chunks,
            'metadata': {
                'num_sources': len(chunks),
                'products': list(set([c['metadata']['product_category'] for c in chunks])),
                'timestamp': datetime.now().isoformat(),
                'k_used': k or self.k,
                'llm_used': use_llm
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'total_chunks': len(self.metadata),
            'embedding_dimension': self.dim,
            'model_name': self.model_name,
            'index_type': 'IndexFlatIP',
            'k_default': self.k,
            'product_distribution': self._get_product_distribution()
        }
    
    def _get_product_distribution(self) -> Dict[str, int]:
        """Get product distribution in the vector store."""
        products = [chunk['product_category'] for chunk in self.metadata]
        distribution = {}
        for product in set(products):
            distribution[product] = products.count(product)
        return distribution


class RAGEvaluator:
    """
    Evaluation tools for RAG pipeline.
    
    Provides qualitative and quantitative evaluation methods.
    """
    
    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize evaluator.
        
        Args:
            rag_pipeline: RAGPipeline instance to evaluate
        """
        self.rag = rag_pipeline
    
    def evaluate_retrieval(self, 
                           test_questions: List[str],
                           k_values: List[int] = [3, 5, 10]) -> pd.DataFrame:
        """
        Evaluate retrieval quality across different k values.
        
        Args:
            test_questions: List of test questions
            k_values: List of k values to test
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        for k in k_values:
            for question in test_questions:
                chunks = self.rag.retrieve(question, k=k)
                
                results.append({
                    'question': question[:50] + '...' if len(question) > 50 else question,
                    'k': k,
                    'num_retrieved': len(chunks),
                    'avg_score': np.mean([c['score'] for c in chunks]) if chunks else 0,
                    'max_score': max([c['score'] for c in chunks]) if chunks else 0,
                    'products': list(set([c['metadata']['product_category'] for c in chunks])),
                    'has_credit_card': any(c['metadata']['product_category'] == 'Credit Card' for c in chunks),
                    'has_loan': any(c['metadata']['product_category'] == 'Personal Loan' for c in chunks),
                })
        
        return pd.DataFrame(results)
    
    def qualitative_evaluation(self, 
                              test_questions: List[Tuple[str, str, str]]) -> pd.DataFrame:
        """
        Qualitative evaluation with scoring.
        
        Args:
            test_questions: List of (question, expected_product, expected_topic)
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        for question, expected_product, expected_topic in test_questions:
            result = self.rag.query(question)
            
            # Score relevance
            relevance_score = self._score_relevance(result, expected_product, expected_topic)
            
            results.append({
                'question': question[:100] + '...' if len(question) > 100 else question,
                'expected_product': expected_product,
                'expected_topic': expected_topic,
                'retrieved_product': result['sources'][0]['metadata']['product_category'] if result['sources'] else 'None',
                'answer_length': len(result['answer']),
                'num_sources': len(result['sources']),
                'relevance_score': relevance_score,
                'answer': result['answer'][:200] + '...' if len(result['answer']) > 200 else result['answer']
            })
        
        return pd.DataFrame(results)
    
    def _score_relevance(self, 
                         result: Dict[str, Any], 
                         expected_product: str,
                         expected_topic: str) -> int:
        """
        Score the relevance of retrieval and answer.
        
        Args:
            result: Query result from RAG pipeline
            expected_product: Expected product category
            expected_topic: Expected topic
            
        Returns:
            Score 1-5
        """
        score = 0
        
        # Check product match
        if result['sources']:
            products = [s['metadata']['product_category'] for s in result['sources']]
            if expected_product in products:
                score += 2
            
            # Check topic match (simple text matching)
            answer_text = result['answer'].lower()
            if expected_topic.lower() in answer_text:
                score += 2
            
            # Check for evidence
            if len(result['sources']) >= 2:
                score += 1
        
        return min(score, 5)  # Cap at 5
    
    def generate_evaluation_report(self, 
                                  test_questions: List[Tuple[str, str, str]]) -> Dict[str, Any]:
        """
        Generate a complete evaluation report.
        
        Args:
            test_questions: List of (question, expected_product, expected_topic)
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = self.qualitative_evaluation(test_questions)
        
        return {
            'total_questions': len(results),
            'avg_relevance_score': results['relevance_score'].mean(),
            'avg_sources': results['num_sources'].mean(),
            'product_matches': (results['retrieved_product'] == results['expected_product']).sum(),
            'product_match_rate': (results['retrieved_product'] == results['expected_product']).mean(),
            'detailed_results': results.to_dict('records')
        }


# Test questions for evaluation
TEST_QUESTIONS = [
    ("Why are customers unhappy with credit card fees?", "Credit Card", "fees"),
    ("What are the main issues with personal loans?", "Personal Loan", "loan issues"),
    ("Tell me about savings account complaints", "Savings Account", "savings"),
    ("What problems do people have with money transfers?", "Money Transfer", "transfer"),
    ("Compare complaints between credit cards and personal loans", "Credit Card", "comparison"),
    ("Are there any issues with customer service?", "Credit Card", "customer service"),
    ("What are the most common complaint types?", "Credit Card", "common complaints"),
    ("How do complaints vary by product?", "Credit Card", "variation"),
]


if __name__ == "__main__":
    # Initialize RAG
    print("Initializing RAG Pipeline...")
    rag = RAGPipeline(k=5)
    
    # Get stats
    print("\n" + "="*60)
    print("VECTOR STORE STATISTICS")
    print("="*60)
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test queries
    print("\n" + "="*60)
    print("TESTING QUERIES")
    print("="*60)
    
    test_queries = [
        "Why are customers unhappy with credit card fees?",
        "What are the main issues with personal loans?",
        "Tell me about savings account complaints"
    ]
    
    for query in test_queries:
        print(f"\n❓ Question: {query}")
        result = rag.query(query, k=3)
        print(f"\n📝 Answer: {result['answer'][:300]}...")
        print(f"\n📎 Sources: {len(result['sources'])} chunks retrieved")
        print(f"   Products: {result['metadata']['products']}")
    
    # Qualitative Evaluation
    print("\n" + "="*60)
    print("QUALITATIVE EVALUATION")
    print("="*60)
    
    evaluator = RAGEvaluator(rag)
    eval_report = evaluator.generate_evaluation_report(TEST_QUESTIONS[:5])
    
    print(f"Total Questions: {eval_report['total_questions']}")
    print(f"Average Relevance Score: {eval_report['avg_relevance_score']:.2f}/5")
    print(f"Average Sources Retrieved: {eval_report['avg_sources']:.2f}")
    print(f"Product Match Rate: {eval_report['product_match_rate']*100:.1f}%")
    
    print("\nDetailed Results:")
    for i, result in enumerate(eval_report['detailed_results']):
        print(f"\n{i+1}. Question: {result['question'][:60]}...")
        print(f"   Expected: {result['expected_product']} - {result['expected_topic']}")
        print(f"   Retrieved: {result['retrieved_product']}")
        print(f"   Relevance Score: {result['relevance_score']}/5")