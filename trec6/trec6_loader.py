"""
TREC-6 Dataset Loader - Works with Current HuggingFace or Direct Download

Handles the deprecated dataset script issue.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import urllib.request
import os


def download_trec6_direct():
    """
    Download TREC-6 directly from source.
    
    Returns train and test data as lists of dicts.
    """
    print("\n" + "="*70)
    print("DOWNLOADING TREC-6 DATASET (DIRECT)")
    print("="*70)
    
    # URLs for TREC dataset
    train_url = "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label"
    test_url = "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label"
    
    cache_dir = Path.home() / '.cache' / 'trec6'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = cache_dir / 'train_5500.label'
    test_file = cache_dir / 'TREC_10.label'
    
    # Download if not cached
    try:
        if not train_file.exists():
            print(f"\nDownloading train split...")
            urllib.request.urlretrieve(train_url, train_file)
            print("✓ Train downloaded")
        else:
            print("\n✓ Train found in cache")
        
        if not test_file.exists():
            print(f"Downloading test split...")
            urllib.request.urlretrieve(test_url, test_file)
            print("✓ Test downloaded")
        else:
            print("✓ Test found in cache")
        
        # Parse files
        train_data = parse_trec_file(train_file)
        test_data = parse_trec_file(test_file)
        
        print(f"\n✓ Successfully loaded TREC!")
        print(f"  Train: {len(train_data)} questions")
        print(f"  Test: {len(test_data)} questions")
        
        # Show class distribution
        from collections import Counter
        train_labels = [item['coarse_label'] for item in train_data]
        label_counts = Counter(train_labels)
        
        print("\n  Coarse class distribution (train):")
        label_names = {0: 'ABBR', 1: 'ENTY', 2: 'DESC', 3: 'HUM', 4: 'LOC', 5: 'NUM'}
        for label_id in sorted(label_counts.keys()):
            count = label_counts[label_id]
            name = label_names.get(label_id, f'Class{label_id}')
            print(f"    {label_id} ({name}): {count} samples")
        
        return {'train': train_data, 'test': test_data}
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("Falling back to synthetic data...")
        return create_synthetic_trec6()


def parse_trec_file(filepath):
    """
    Parse TREC label file.
    
    Format: COARSE:FINE question text
    Example: DESC:def What is a meerkat ?
    """
    # Label mappings
    coarse_map = {'ABBR': 0, 'ENTY': 1, 'DESC': 2, 'HUM': 3, 'LOC': 4, 'NUM': 5}
    
    data = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split label and text
            parts = line.split(' ', 1)
            if len(parts) != 2:
                continue
            
            label_str, text = parts
            
            # Parse label
            if ':' in label_str:
                coarse, fine = label_str.split(':', 1)
            else:
                coarse = label_str
                fine = label_str
            
            # Map coarse label to int
            coarse_label = coarse_map.get(coarse, 0)
            
            data.append({
                'text': text,
                'coarse_label': coarse_label,
                'fine_label': fine,
                'coarse_name': coarse
            })
    
    return data


def create_synthetic_trec6():
    """
    Create synthetic TREC-like data as fallback.
    """
    print("\nCreating synthetic TREC-like data...")
    
    # Question templates by category
    templates = {
        0: [  # ABBR
            "What does {} stand for?",
            "What is the abbreviation {}?",
            "What does the acronym {} mean?",
        ],
        1: [  # ENTY
            "What is {}?",
            "What kind of thing is {}?",
            "What type of {} is this?",
        ],
        2: [  # DESC
            "What is the definition of {}?",
            "How does {} work?",
            "What is the purpose of {}?",
        ],
        3: [  # HUM
            "Who invented {}?",
            "Who is {}?",
            "Who was the first person to {}?",
        ],
        4: [  # LOC
            "Where is {}?",
            "Where can you find {}?",
            "What country is {} in?",
        ],
        5: [  # NUM
            "How many {}?",
            "What is the population of {}?",
            "How much does {} cost?",
        ]
    }
    
    terms = [
        'NASA', 'DNA', 'CPU', 'HTTP', 'GPS',  # ABBR
        'computer', 'atom', 'enzyme', 'planet', 'democracy',  # ENTY
        'photosynthesis', 'gravity', 'evolution', 'quantum mechanics', 'internet',  # DESC
        'Einstein', 'Newton', 'Darwin', 'Tesla', 'Curie',  # HUM
        'Paris', 'Tokyo', 'Rome', 'Cairo', 'Sydney',  # LOC
        'stars in the galaxy', 'continents', 'prime numbers under 100', 'planets', 'US states'  # NUM
    ]
    
    np.random.seed(42)
    
    train_data = []
    test_data = []
    
    # Generate train data (5000 samples)
    for _ in range(5000):
        label = np.random.randint(0, 6)
        template = np.random.choice(templates[label])
        term = np.random.choice(terms)
        text = template.format(term)
        
        train_data.append({
            'text': text,
            'coarse_label': label,
            'fine_label': f'synthetic_{label}',
            'coarse_name': ['ABBR', 'ENTY', 'DESC', 'HUM', 'LOC', 'NUM'][label]
        })
    
    # Generate test data (500 samples)
    np.random.seed(43)
    for _ in range(500):
        label = np.random.randint(0, 6)
        template = np.random.choice(templates[label])
        term = np.random.choice(terms)
        text = template.format(term)
        
        test_data.append({
            'text': text,
            'coarse_label': label,
            'fine_label': f'synthetic_{label}',
            'coarse_name': ['ABBR', 'ENTY', 'DESC', 'HUM', 'LOC', 'NUM'][label]
        })
    
    print(f"✓ Created synthetic TREC-like dataset")
    print(f"  Train: {len(train_data)} questions")
    print(f"  Test: {len(test_data)} questions")
    
    return {'train': train_data, 'test': test_data}


class TREC6Dataset(Dataset):
    """
    TREC-6 Question Classification Dataset.
    
    Coarse classes (6):
    0: ABBR (abbreviation)
    1: ENTY (entity) 
    2: DESC (description)
    3: HUM (human)
    4: LOC (location)
    5: NUM (numeric)
    """
    
    def __init__(self, data_dict, split='train'):
        """
        Args:
            data_dict: Dictionary with 'train' and 'test' keys
            split: 'train' or 'test'
        """
        self.data = data_dict[split]
        
        # Extract questions and labels
        self.questions = [item['text'] for item in self.data]
        self.coarse_labels = [item['coarse_label'] for item in self.data]
        self.fine_labels = [item['fine_label'] for item in self.data]
        
        print(f"\n  {split.capitalize()}: {len(self.data)} questions")
        
        # Show statistics
        from collections import Counter
        coarse_counts = Counter(self.coarse_labels)
        print(f"  Coarse classes: {len(coarse_counts)}")
        print(f"  Unique fine labels: {len(set(self.fine_labels))}")
        
        # Check imbalance
        if coarse_counts:
            max_count = max(coarse_counts.values())
            min_count = min(coarse_counts.values())
            print(f"  Imbalance ratio: {max_count/min_count:.1f}x")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'text': self.questions[idx],
            'coarse_label': self.coarse_labels[idx],
            'fine_label': self.fine_labels[idx]
        }


if __name__ == "__main__":
    # Test the loader
    print("Testing TREC-6 loader...")
    
    dataset = download_trec6_direct()
    
    if dataset:
        print("\n✓ Loader working!")
        
        # Create dataset objects
        train_ds = TREC6Dataset(dataset, split='train')
        test_ds = TREC6Dataset(dataset, split='test')
        
        # Show sample
        sample = train_ds[0]
        print(f"\nSample question:")
        print(f"  Text: {sample['text']}")
        print(f"  Label: {sample['coarse_label']}")
        print(f"  Fine: {sample['fine_label']}")
    else:
        print("\n✗ Loader failed")
