import os
import requests
import numpy as np

def download_file(url, filename):
    """Download a file from a URL to a local file."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def prepare_enwik8(input_file, train_file, val_file, test_file):
    """Process the enwik8 dataset and create train, validation, and test sets."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    
    # Create a vocabulary (unique characters)
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    
    # Encode the entire dataset
    data_encoded = np.array([stoi[ch] for ch in data], dtype=np.uint16)
    
    # Split the data
    n1 = 90_000_000  # first 90M characters for train
    n2 = 95_000_000  # first 95M characters for train+val
    train_data = data_encoded[:n1]
    val_data = data_encoded[n1:n2]
    test_data = data_encoded[n2:]
    
    # Save the encoded datasets
    train_data.tofile(train_file)
    val_data.tofile(val_file)
    test_data.tofile(test_file)
    
    # Save the vocabulary
    with open(os.path.join(os.path.dirname(train_file), 'vocab.txt'), 'w') as f:
        for ch in chars:
            f.write(ch + '\n')

    return vocab_size

if __name__ == '__main__':
    # URL of the enwik8 dataset
    url = 'https://data.deepai.org/enwik8.zip'
    
    # Local file names
    zip_file = 'enwik8.zip'
    input_file = 'enwik8'
    train_file = 'train.bin'
    val_file = 'val.bin'
    test_file = 'test.bin'
    
    # Download the dataset if it doesn't exist
    if not os.path.exists(input_file):
        print(f"Downloading {url}...")
        download_file(url, zip_file)
        
        # Unzip the file
        import zipfile
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall()
        
        # Remove the zip file
        os.remove(zip_file)
    
    # Process the dataset
    vocab_size = prepare_enwik8(input_file, train_file, val_file, test_file)
    
    print(f"Prepared enwik8 dataset:")
    print(f"Vocab size: {vocab_size}")
    print(f"Train data: {os.path.getsize(train_file)} bytes")
    print(f"Val data: {os.path.getsize(val_file)} bytes")
    print(f"Test data: {os.path.getsize(test_file)} bytes")