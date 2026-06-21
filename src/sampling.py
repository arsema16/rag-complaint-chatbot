"""
Stratified Sampling Module for Complaint Data

This module creates a stratified sample of 10,000-15,000 complaints
with proportional representation across all product categories.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StratifiedSampler:
    """
    Handles stratified sampling of complaint data with proportional
    representation across product categories.
    """
    
    def __init__(self, 
                 data_path: str = 'data/raw/complaints.csv',
                 sample_size: int = 15000,
                 random_state: int = 42,
                 min_samples_per_product: int = 100):
        """
        Initialize the stratified sampler.
        
        Args:
            data_path: Path to the raw complaints CSV file
            sample_size: Target total sample size (10,000-15,000 recommended)
            random_state: Random seed for reproducibility
            min_samples_per_product: Minimum samples per product category
        """
        self.data_path = data_path
        self.sample_size = sample_size
        self.random_state = random_state
        self.min_samples_per_product = min_samples_per_product
        self.df = None
        self.sampled_df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the complaint data from CSV."""
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path, low_memory=False)
        logger.info(f"Loaded {len(self.df):,} records with {len(self.df.columns)} columns")
        return self.df
    
    def map_product_category(self, product: str) -> str:
        """
        Map product names to the four target categories.
        
        Args:
            product: Product name string
            
        Returns:
            Mapped product category
        """
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
        else:
            return 'Other'
    
    def add_product_categories(self) -> pd.DataFrame:
        """Add product category column to the dataset."""
        logger.info("Mapping products to categories...")
        self.df['Product_Category'] = self.df['Product'].apply(self.map_product_category)
        
        # Show distribution
        distribution = self.df['Product_Category'].value_counts()
        logger.info(f"Product distribution:\n{distribution}")
        
        return self.df
    
    def calculate_sample_sizes(self) -> Dict[str, int]:
        """
        Calculate sample sizes for each product category using proportional allocation.
        
        Returns:
            Dictionary mapping product categories to sample sizes
        """
        # Filter to target products
        target_products = ['Credit Card', 'Personal Loan', 'Savings Account', 'Money Transfer']
        target_df = self.df[self.df['Product_Category'].isin(target_products)]
        
        # Calculate proportions
        product_counts = target_df['Product_Category'].value_counts()
        total_target = len(target_df)
        proportions = product_counts / total_target
        
        logger.info(f"Product proportions:\n{proportions}")
        
        # Calculate initial sample sizes
        sample_sizes = (proportions * self.sample_size).round().astype(int)
        
        # Ensure minimum samples per product
        sample_sizes = sample_sizes.clip(lower=self.min_samples_per_product)
        
        # Adjust to match target sample size
        total_samples = sample_sizes.sum()
        
        if total_samples > self.sample_size:
            # Reduce from largest categories
            while total_samples > self.sample_size:
                max_prod = sample_sizes.idxmax()
                if sample_sizes[max_prod] > self.min_samples_per_product:
                    sample_sizes[max_prod] -= 1
                    total_samples = sample_sizes.sum()
                else:
                    break
        elif total_samples < self.sample_size:
            # Add to largest categories
            while total_samples < self.sample_size:
                max_prod = sample_sizes.idxmax()
                sample_sizes[max_prod] += 1
                total_samples = sample_sizes.sum()
        
        logger.info(f"Final sample sizes per product:\n{sample_sizes}")
        logger.info(f"Total sample size: {total_samples}")
        
        return sample_sizes.to_dict()
    
    def perform_stratified_sampling(self) -> pd.DataFrame:
        """
        Perform stratified sampling based on product categories.
        
        Returns:
            Sampled DataFrame with proportional representation
        """
        logger.info("Performing stratified sampling...")
        
        target_products = ['Credit Card', 'Personal Loan', 'Savings Account', 'Money Transfer']
        sample_sizes = self.calculate_sample_sizes()
        
        sampled_dfs = []
        
        for product, n_samples in sample_sizes.items():
            # Get product data
            product_df = self.df[self.df['Product_Category'] == product]
            
            if len(product_df) >= n_samples:
                # Sample with replacement if enough data
                sample = product_df.sample(n=n_samples, 
                                         random_state=self.random_state,
                                         replace=False)
                logger.info(f"Sampled {n_samples:,} from {product} ({len(product_df):,} available)")
            else:
                # Use all available if not enough
                sample = product_df
                logger.warning(f"Not enough samples for {product}: {len(product_df)} available, {n_samples} needed")
                logger.info(f"Using all {len(product_df)} samples from {product}")
            
            sampled_dfs.append(sample)
        
        # Combine samples
        self.sampled_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Verify final distribution
        final_distribution = self.sampled_df['Product_Category'].value_counts()
        logger.info(f"Final sampled distribution:\n{final_distribution}")
        logger.info(f"Total samples: {len(self.sampled_df):,}")
        
        return self.sampled_df
    
    def save_sample(self, output_path: str = 'data/sampled_complaints.csv'):
        """Save the sampled dataset to CSV."""
        if self.sampled_df is None:
            raise ValueError("No sample available. Run perform_stratified_sampling() first.")
        
        logger.info(f"Saving sample to {output_path}")
        self.sampled_df.to_csv(output_path, index=False)
        
        # Also save sampling metadata
        metadata = {
            'sample_size': len(self.sampled_df),
            'random_state': self.random_state,
            'product_distribution': self.sampled_df['Product_Category'].value_counts().to_dict()
        }
        
        import json
        with open('data/sampling_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved sampling metadata to data/sampling_metadata.json")
        
    def get_sampling_report(self) -> Dict:
        """Generate a detailed sampling report."""
        if self.sampled_df is None:
            return {"status": "No sample created yet"}
        
        return {
            "total_samples": len(self.sampled_df),
            "product_distribution": self.sampled_df['Product_Category'].value_counts().to_dict(),
            "product_percentages": (self.sampled_df['Product_Category'].value_counts(normalize=True) * 100).to_dict(),
            "random_seed": self.random_state,
            "target_sample_size": self.sample_size,
            "min_samples_per_product": self.min_samples_per_product
        }

def create_stratified_sample(data_path: str = 'data/raw/complaints.csv',
                            sample_size: int = 15000,
                            output_path: str = 'data/sampled_complaints.csv',
                            random_state: int = 42) -> pd.DataFrame:
    """
    Convenience function to create a stratified sample.
    
    Args:
        data_path: Path to raw complaint data
        sample_size: Target sample size (10,000-15,000 recommended)
        output_path: Path to save the sample
        random_state: Random seed for reproducibility
    
    Returns:
        Sampled DataFrame
    """
    sampler = StratifiedSampler(
        data_path=data_path,
        sample_size=sample_size,
        random_state=random_state
    )
    
    # Load data
    sampler.load_data()
    
    # Add product categories
    sampler.add_product_categories()
    
    # Perform sampling
    sampled_df = sampler.perform_stratified_sampling()
    
    # Save sample
    sampler.save_sample(output_path)
    
    # Print report
    report = sampler.get_sampling_report()
    print("\n" + "="*60)
    print("SAMPLING REPORT")
    print("="*60)
    print(f"Total Samples: {report['total_samples']:,}")
    print(f"Target Sample Size: {report['target_sample_size']:,}")
    print(f"Random Seed: {report['random_seed']}")
    print("\nProduct Distribution:")
    for product, count in report['product_distribution'].items():
        percentage = report['product_percentages'][product]
        print(f"  {product}: {count:,} ({percentage:.1f}%)")
    
    return sampled_df

if __name__ == "__main__":
    # Run sampling with default parameters
    sampled_df = create_stratified_sample(
        data_path='data/raw/complaints.csv',
        sample_size=15000,
        output_path='data/sampled_complaints.csv',
        random_state=42
    )