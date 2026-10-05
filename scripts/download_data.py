import urllib.request
from pathlib import Path

def main():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"
    dest = Path("data/raw/heart_failure.csv")
    
    if not dest.exists():
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, dest)
        print("Download complete.")
    else:
        print("File already exists.")

if __name__ == "__main__":
    main()