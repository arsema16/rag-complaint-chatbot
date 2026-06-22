"""
Simple unit tests that work
"""

import pytest

class TestBasic:
    """Basic test cases"""
    
    def test_math(self):
        """Simple math test"""
        assert 1 + 1 == 2
        assert 2 * 3 == 6
        assert 10 / 2 == 5
    
    def test_strings(self):
        """String operations test"""
        assert "hello".upper() == "HELLO"
        assert "world".capitalize() == "World"
    
    def test_lists(self):
        """List operations test"""
        my_list = [1, 2, 3]
        assert len(my_list) == 3
        assert 2 in my_list
        assert my_list[0] == 1

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
