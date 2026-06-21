"""
Data loader module for CFPB complaint dataset
"""

import pandas as pd
import os
import zipfile
import urllib.request
from pathlib import Path

class DataLoader:
    """
    Handles downloading and loading of CFPB complaint data
    """
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_data(self):
        url = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
        zip_path = self.data_dir / "complaints.csv.zip"
        
        print(f"Downloading data from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded to {zip_path}")
        
        print("Extracting data...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        print("Extraction complete")
        
        csv_files = list(self.data_dir.glob("*.csv"))
        if csv_files:
            return csv_files[0]
        return None
    
    def load_data(self, sample_size=None):
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            print("No CSV file found. Downloading...")
            csv_path = self.download_data()
            if csv_path:
                csv_files = [csv_path]
            else:
                raise FileNotFoundError("Could not find or download data")
        
        csv_path = csv_files[0]
        print(f"Loading data from {csv_path}")
        
        if sample_size:
            df = pd.read_csv(csv_path, nrows=sample_size, low_memory=False)
        else:
            df = pd.read_csv(csv_path, low_memory=False)
            
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_data(sample_size=10000)
    print(df.head())