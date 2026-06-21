"""
Download CFPB data
"""

import urllib.request
import zipfile
import os

print("="*60)
print("Downloading CFPB Complaint Data")
print("="*60)

os.makedirs('data/raw', exist_ok=True)

url = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
zip_path = "data/raw/complaints.csv.zip"

print(f"\nDownloading from: {url}")
urllib.request.urlretrieve(url, zip_path)
print(f"Downloaded to: {zip_path}")

print("\nExtracting...")
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall("data/raw")

print("\n✅ Done! Data extracted to: data/raw/")
print("Files in data/raw:")
for f in os.listdir('data/raw'):
    if f.endswith('.csv'):
        size = os.path.getsize(f'data/raw/{f}') / (1024*1024)
        print(f"  - {f} ({size:.1f} MB)")