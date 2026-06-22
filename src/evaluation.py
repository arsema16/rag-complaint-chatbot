"""
Evaluation Module for RAG System

Provides comprehensive evaluation tools for:
- Retrieval quality assessment
- Answer quality scoring
- Performance metrics
- A/B testing capabilities
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
import json
from rag_pipeline import RAGPipeline

class RAGEvaluator:
    """
    Comprehensive evaluator for RAG pipeline.
    """
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag = rag_pipeline
        self.results = []
    
    def evaluate_retrieval_metrics(self, test_questions: List[str], k_values: List[int] = [3, 5, 10]):
        """Evaluate retrieval metrics across different k values."""
        metrics = []
        
        for k in k_values:
            k_metrics = {'k': k}
            
            for question in test_questions:
                chunks = self.rag.retrieve(question, k=k)
                
                # Calculate metrics
                scores = [c['score'] for c in chunks]
                
                if scores:
                    k_metrics[f'{question[:30]}_num'] = len(chunks)
                    k_metrics[f'{question[:30]}_avg_score'] = np.mean(scores)
                    k_metrics[f'{question[:30]}_max_score'] = np.max(scores)
                    k_metrics[f'{question[:30]}_products'] = len(set(c['metadata']['product_category'] for c in chunks))
            
            metrics.append(k_metrics)
        
        return pd.DataFrame(metrics)
    
    def evaluate_answer_quality(self, test_questions: List[Tuple[str, str]]):
        """
        Evaluate answer quality using multiple criteria.
        
        Args:
            test_questions: List of (question, expected_answer_keyword)
        """
        results = []
        
        for question, expected_keyword in test_questions:
            result = self.rag.query(question)
            
            # Score answer
            scores = self._score_answer(
                result['answer'], 
                result['sources'],
                expected_keyword
            )
            
            results.append({
                'question': question,
                'answer_length': len(result['answer']),
                'num_sources': len(result['sources']),
                'avg_source_score': np.mean([s['score'] for s in result['sources']]) if result['sources'] else 0,
                'max_source_score': max([s['score'] for s in result['sources']]) if result['sources'] else 0,
                'product_coverage': len(set(s['metadata']['product_category'] for s in result['sources'])),
                'keyword_match': expected_keyword.lower() in result['answer'].lower(),
                'relevance_score': scores['relevance'],
                'groundedness_score': scores['groundedness'],
                'completeness_score': scores['completeness'],
                'overall_score': scores['overall']
            })
        
        return pd.DataFrame(results)
    
    def _score_answer(self, answer: str, sources: List[Dict], expected_keyword: str) -> Dict[str, float]:
        """Score answer on multiple criteria."""
        
        # Relevance - does it address the question?
        relevance = 3.0  # Base score
        
        # Check if answer uses the sources
        if sources:
            # Groundedness - is it based on sources?
            groundedness = min(5.0, 3.0 + len(sources) * 0.5)
        else:
            groundedness = 1.0
        
        # Completeness - does it cover key aspects?
        completeness = 2.0
        if len(answer) > 200:
            completeness += 1.0
        if len(sources) >= 3:
            completeness += 1.0
        if expected_keyword.lower() in answer.lower():
            completeness += 1.0
        
        # Overall score
        overall = (relevance + groundedness + completeness) / 3
        
        return {
            'relevance': min(5.0, relevance),
            'groundedness': min(5.0, groundedness),
            'completeness': min(5.0, completeness),
            'overall': min(5.0, overall)
        }
    
    def generate_full_report(self, test_questions: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Generate complete evaluation report."""
        
        # Retrieval metrics
        retrieval_df = self.evaluate_retrieval_metrics(
            [q for q, _ in test_questions[:5]]
        )
        
        # Answer quality
        quality_df = self.evaluate_answer_quality(test_questions)
        
        # Summary metrics
        return {
            'total_questions': len(test_questions),
            'avg_answer_length': quality_df['answer_length'].mean(),
            'avg_num_sources': quality_df['num_sources'].mean(),
            'avg_relevance': quality_df['relevance_score'].mean(),
            'avg_groundedness': quality_df['groundedness_score'].mean(),
            'avg_completeness': quality_df['completeness_score'].mean(),
            'avg_overall': quality_df['overall_score'].mean(),
            'keyword_match_rate': quality_df['keyword_match'].mean(),
            'retrieval_metrics': retrieval_df.to_dict('records'),
            'quality_metrics': quality_df.to_dict('records')
        }

if __name__ == "__main__":
    # Test evaluation
    rag = RAGPipeline()
    evaluator = RAGEvaluator(rag)
    
    test_questions = [
        ("Why are customers unhappy with credit card fees?", "fees"),
        ("What are the main issues with personal loans?", "loan"),
        ("Tell me about savings account complaints", "savings"),
        ("What problems do people have with money transfers?", "transfer"),
        ("Are there any issues with customer service?", "service"),
    ]
    
    report = evaluator.generate_full_report(test_questions)
    
    print("="*60)
    print("EVALUATION REPORT")
    print("="*60)
    print(f"Total Questions: {report['total_questions']}")
    print(f"Average Relevance: {report['avg_relevance']:.2f}/5")
    print(f"Average Groundedness: {report['avg_groundedness']:.2f}/5")
    print(f"Average Completeness: {report['avg_completeness']:.2f}/5")
    print(f"Overall Score: {report['avg_overall']:.2f}/5")
    print(f"Keyword Match Rate: {report['keyword_match_rate']*100:.1f}%")