"""
Stratified Sampling Module
"""

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StratifiedSampler:
    def __init__(self, data_path=None, sample_size=15000, random_state=42):
        self.data_path = data_path
        self.sample_size = sample_size
        self.random_state = random_state
        self.df = None
        self.sampled_df = None
    
    def load_data(self):
        logger.info("Loading data...")
        self.df = pd.read_csv(self.data_path, low_memory=False)
        return self.df
    
    def map_product_category(self, product):
        if pd.isna(product):
            return 'Other'
        product_lower = str(product).lower()
        if 'credit card' in product_lower:
            return 'Credit Card'
        elif 'loan' in product_lower or 'payday' in product_lower:
            return 'Personal Loan'
        elif 'savings' in product_lower or 'checking' in product_lower:
            return 'Savings Account'
        elif 'money' in product_lower or 'transfer' in product_lower:
            return 'Money Transfer'
        return 'Other'
    
    def add_product_categories(self):
        self.df['Product_Category'] = self.df['Product'].apply(self.map_product_category)
        return self.df
    
    def perform_stratified_sampling(self):
        logger.info("Performing stratified sampling...")
        target_products = ['Credit Card', 'Personal Loan', 'Savings Account', 'Money Transfer']
        product_counts = self.df[self.df['Product_Category'].isin(target_products)]['Product_Category'].value_counts()
        proportions = product_counts / product_counts.sum()
        sample_sizes = (proportions * self.sample_size).round().astype(int)
        sample_sizes = sample_sizes.clip(lower=100)
        
        sampled_dfs = []
        for product, n_samples in sample_sizes.items():
            product_df = self.df[self.df['Product_Category'] == product]
            if len(product_df) >= n_samples:
                sample = product_df.sample(n=n_samples, random_state=self.random_state)
            else:
                sample = product_df
            sampled_dfs.append(sample)
        
        self.sampled_df = pd.concat(sampled_dfs, ignore_index=True)
        logger.info(f"Created sample with {len(self.sampled_df)} records")
        return self.sampled_df

def create_stratified_sample(data_path=None, sample_size=15000, output_path=None, random_state=42):
    sampler = StratifiedSampler(data_path, sample_size, random_state)
    if data_path:
        sampler.load_data()
        sampler.add_product_categories()
        return sampler.perform_stratified_sampling()
    return pd.DataFrame()
