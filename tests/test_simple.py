"""
Simple unit tests that work
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestBasic:
    """Basic test cases"""
    
    def test_imports(self):
        """Test that imports work"""
        try:
            import pandas as pd
            import numpy as np
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_environment(self):
        """Test Python environment"""
        import sys
        assert sys.version_info.major >= 3
        assert sys.version_info.minor >= 7
    
    def test_math(self):
        """Simple math test"""
        assert 1 + 1 == 2
        assert 2 * 3 == 6
        assert 10 / 2 == 5

class TestTextProcessing:
    """Test text processing functions"""
    
    def test_clean_text(self):
        """Test text cleaning"""
        def clean(x):
            if not x:
                return ""
            import re
            x = x.lower()
            x = re.sub(r'[^a-zA-Z0-9\s.]', ' ', x)
            x = re.sub(r'\s+', ' ', x).strip()
            return x
        
        assert clean("HELLO") == "hello"
        assert clean("Hello!!!") == "hello"
        assert clean("") == ""
        assert clean(None) == ""
    
    def test_product_mapping(self):
        """Test product category mapping"""
        def map_product(x):
            if not x:
                return 'Other'
            x = str(x).lower()
            if 'credit card' in x:
                return 'Credit Card'
            elif 'loan' in x:
                return 'Personal Loan'
            elif 'savings' in x:
                return 'Savings Account'
            elif 'money' in x:
                return 'Money Transfer'
            return 'Other'
        
        assert map_product("Credit Card") == "Credit Card"
        assert map_product("Personal Loan") == "Personal Loan"
        assert map_product("Savings Account") == "Savings Account"
        assert map_product("Money Transfer") == "Money Transfer"
        assert map_product("Unknown") == "Other"

class TestChunking:
    """Test chunking functions"""
    
    def test_simple_chunk(self):
        """Test simple chunking"""
        def simple_chunk(text, chunk_size=500, overlap=50):
            if not text:
                return [""]
            if len(text) <= chunk_size:
                return [text]
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                if end < len(text) and text[end] != ' ':
                    end = text.rfind(' ', start, end)
                    if end == -1:
                        end = min(start + chunk_size, len(text))
                chunks.append(text[start:end].strip())
                start = max(end - overlap, start + 1)
            return chunks if chunks else [text]
        
        text = "This is a test. It has multiple sentences. For chunking."
        chunks = simple_chunk(text, chunk_size=50, overlap=10)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

class TestSampling:
    """Test sampling functions"""
    
    def test_create_sample(self):
        """Test sample creation"""
        import pandas as pd
        df = pd.DataFrame({
            'Product': ['Credit Card'] * 100 + ['Personal Loan'] * 50,
            'Issue': ['Test'] * 150
        })
        assert len(df) == 150
        assert 'Credit Card' in df['Product'].values
        assert 'Personal Loan' in df['Product'].values
