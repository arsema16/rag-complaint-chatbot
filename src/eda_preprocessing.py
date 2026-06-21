"""
EDA and Preprocessing for CFPB Complaint Data - OPTIMIZED VERSION
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load data with minimal processing"""
    print("="*60)
    print("STEP 1: Loading Data")
    print("="*60)
    
    # Load only necessary columns to save memory
    columns_to_load = ['Complaint ID', 'Product', 'Issue', 'Sub-issue', 
                       'Company', 'State', 'Date received', 'Consumer complaint narrative']
    
    try:
        df = pd.read_csv('data/raw/complaints.csv', 
                         usecols=[col for col in columns_to_load if col in pd.read_csv('data/raw/complaints.csv', nrows=1).columns],
                         low_memory=False)
        print(f"Loaded {len(df)} records")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except:
        # Fallback: load all columns but limit rows
        print("Loading sample of data...")
        df = pd.read_csv('data/raw/complaints.csv', nrows=50000, low_memory=False)
        print(f"Loaded {len(df)} records (sample)")
        return df

def map_product_category(product):
    """Map products to four target categories"""
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

def clean_text(text):
    """Clean complaint text - SIMPLIFIED for speed"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    """Run the complete EDA pipeline - FAST VERSION"""
    print("CFPB COMPLAINT DATA EDA AND PREPROCESSING (OPTIMIZED)")
    print("="*60)
    
    # 1. Load data (limited rows for speed)
    print("\nLoading data...")
    df = pd.read_csv('data/raw/complaints.csv', nrows=50000, low_memory=False)
    print(f"Loaded {len(df):,} records (first 50,000 for speed)")
    
    # 2. Find narrative column
    narrative_col = 'Consumer complaint narrative' if 'Consumer complaint narrative' in df.columns else 'Issue'
    print(f"\nUsing '{narrative_col}' as narrative column")
    
    # 3. Map products
    print("\n" + "="*60)
    print("STEP 2: Product Distribution")
    print("="*60)
    df['Product_Category'] = df['Product'].apply(map_product_category)
    print(df['Product_Category'].value_counts())
    
    # Filter to target products
    df_filtered = df[df['Product_Category'] != 'Other'].copy()
    print(f"\nFiltered dataset: {len(df_filtered):,} records")
    
    # 4. Narrative analysis
    print("\n" + "="*60)
    print("STEP 3: Narrative Analysis")
    print("="*60)
    df_filtered['has_narrative'] = df_filtered[narrative_col].notna()
    df_filtered['narrative_length'] = df_filtered[narrative_col].fillna('').astype(str).str.len()
    
    has_narrative = df_filtered['has_narrative'].sum()
    print(f"Complaints with narratives: {has_narrative:,}")
    if len(df_filtered) > 0:
        print(f"Percentage: {has_narrative/len(df_filtered)*100:.2f}%")
    
    # 5. Clean text (only on filtered data with narratives)
    print("\n" + "="*60)
    print("STEP 4: Text Cleaning")
    print("="*60)
    df_with_narratives = df_filtered[df_filtered['has_narrative']].copy()
    print(f"Cleaning {len(df_with_narratives):,} narratives...")
    df_with_narratives['cleaned_narrative'] = df_with_narratives[narrative_col].apply(clean_text)
    df_with_narratives['cleaned_length'] = df_with_narratives['cleaned_narrative'].str.len()
    
    # Filter to non-empty
    df_final = df_with_narratives[df_with_narratives['cleaned_narrative'].str.len() > 10].copy()
    print(f"After cleaning: {len(df_final):,} records")
    
    # 6. Save
    print("\n" + "="*60)
    print("STEP 5: Saving Data")
    print("="*60)
    columns_to_save = ['Complaint ID', 'Product', 'Product_Category', 'Issue', 
                       'Company', 'cleaned_narrative', 'cleaned_length']
    columns_to_save = [c for c in columns_to_save if c in df_final.columns]
    df_final[columns_to_save].to_csv('data/filtered_complaints.csv', index=False)
    print(f"Saved {len(df_final):,} records to data/filtered_complaints.csv")
    
    # 7. Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Final dataset: {len(df_final):,} complaints")
    if len(df_final) > 0:
        print(f"Products: {df_final['Product_Category'].unique().tolist()}")
        print(f"Avg narrative length: {df_final['cleaned_length'].mean():.0f} chars")

if __name__ == "__main__":
    main()