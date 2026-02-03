#!/usr/bin/env python3
"""
Data Validation Script - Check if TREC-6 dataset is corrupted or problematic

Run this BEFORE training to ensure data is clean.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import Counter
import numpy as np

# Import your data loader
try:
    from trec6_loader import download_trec6_direct, TREC6Dataset
    print("✓ Successfully imported TREC-6 loader\n")
except Exception as e:
    print(f"✗ Failed to import loader: {e}")
    sys.exit(1)


def validate_dataset():
    """Comprehensive dataset validation."""
    
    print("="*70)
    print("TREC-6 DATASET VALIDATION")
    print("="*70)
    print()
    
    # Load data
    try:
        dataset_dict = download_trec6_direct()
        train_ds = TREC6Dataset(dataset_dict, split='train')
        test_ds = TREC6Dataset(dataset_dict, split='test')
        print("\n✓ Dataset loaded successfully\n")
    except Exception as e:
        print(f"\n✗ Failed to load dataset: {e}")
        return False
    
    # === CHECK 1: Dataset size ===
    print("-" * 70)
    print("CHECK 1: Dataset Size")
    print("-" * 70)
    
    train_size = len(train_ds)
    test_size = len(test_ds)
    
    print(f"Train size: {train_size}")
    print(f"Test size: {test_size}")
    
    if train_size < 1000:
        print("⚠️  WARNING: Train set is very small!")
    else:
        print("✓ Train set size looks good")
    
    if test_size < 100:
        print("⚠️  WARNING: Test set is very small!")
    else:
        print("✓ Test set size looks good")
    print()
    
    # === CHECK 2: Label distribution ===
    print("-" * 70)
    print("CHECK 2: Label Distribution")
    print("-" * 70)
    
    train_labels = [train_ds[i]['coarse_label'] for i in range(len(train_ds))]
    test_labels = [test_ds[i]['coarse_label'] for i in range(len(test_ds))]
    
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    
    print("\nTrain label distribution:")
    label_names = {0: 'ABBR', 1: 'ENTY', 2: 'DESC', 3: 'HUM', 4: 'LOC', 5: 'NUM'}
    
    for label_id in sorted(train_counts.keys()):
        count = train_counts[label_id]
        pct = 100 * count / train_size
        name = label_names.get(label_id, f'Class{label_id}')
        print(f"  {label_id} ({name:4s}): {count:4d} ({pct:5.1f}%)")
    
    # Check for severe imbalance
    max_count = max(train_counts.values())
    min_count = min(train_counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f"\nImbalance ratio: {imbalance_ratio:.1f}:1")
    if imbalance_ratio > 10:
        print("⚠️  WARNING: Severe class imbalance!")
    elif imbalance_ratio > 5:
        print("⚠️  Moderate class imbalance (use class weights)")
    else:
        print("✓ Class distribution is reasonable")
    
    # Check for missing classes
    expected_classes = set(range(6))
    actual_classes = set(train_counts.keys())
    missing = expected_classes - actual_classes
    
    if missing:
        print(f"✗ MISSING CLASSES: {missing}")
        return False
    else:
        print("✓ All 6 classes present")
    print()
    
    # === CHECK 3: Text quality ===
    print("-" * 70)
    print("CHECK 3: Text Quality")
    print("-" * 70)
    
    # Sample some texts
    sample_indices = np.random.choice(len(train_ds), min(100, len(train_ds)), replace=False)
    text_lengths = []
    empty_count = 0
    very_short_count = 0
    
    for idx in sample_indices:
        text = train_ds[idx]['text']
        length = len(text)
        text_lengths.append(length)
        
        if length == 0:
            empty_count += 1
        elif length < 5:
            very_short_count += 1
    
    avg_length = np.mean(text_lengths)
    max_length = np.max(text_lengths)
    min_length = np.min(text_lengths)
    
    print(f"Text length statistics (sample of {len(sample_indices)}):")
    print(f"  Average: {avg_length:.1f} characters")
    print(f"  Max: {max_length} characters")
    print(f"  Min: {min_length} characters")
    print(f"  Empty texts: {empty_count}")
    print(f"  Very short (<5 chars): {very_short_count}")
    
    if empty_count > 0:
        print("✗ FOUND EMPTY TEXTS! This will break training!")
        return False
    elif very_short_count > 10:
        print("⚠️  Many very short texts found")
    else:
        print("✓ Text lengths look reasonable")
    print()
    
    # === CHECK 4: Show examples ===
    print("-" * 70)
    print("CHECK 4: Sample Data")
    print("-" * 70)
    
    print("\nRandom samples from each class:")
    for class_id in range(6):
        # Find an example of this class
        for idx in range(len(train_ds)):
            sample = train_ds[idx]
            if sample['coarse_label'] == class_id:
                name = label_names[class_id]
                text = sample['text'][:80] + ('...' if len(sample['text']) > 80 else '')
                print(f"\n  Class {class_id} ({name}):")
                print(f"    \"{text}\"")
                break
    print()
    
    # === CHECK 5: Theoretical max accuracy ===
    print("-" * 70)
    print("CHECK 5: Baseline Accuracy")
    print("-" * 70)
    
    # Random baseline
    random_acc = 100.0 / 6
    print(f"Random guessing: {random_acc:.1f}%")
    
    # Majority class baseline
    majority_class = max(test_counts, key=test_counts.get)
    majority_count = test_counts[majority_class]
    majority_acc = 100.0 * majority_count / test_size
    print(f"Majority class baseline: {majority_acc:.1f}%")
    
    # Expected achievable accuracy
    if imbalance_ratio < 3:
        expected = "90-98%"
    elif imbalance_ratio < 5:
        expected = "85-95%"
    else:
        expected = "80-92%"
    
    print(f"Expected achievable: {expected}")
    print()
    
    # === CHECK 6: Data type validation ===
    print("-" * 70)
    print("CHECK 6: Data Types")
    print("-" * 70)
    
    sample = train_ds[0]
    print(f"Text type: {type(sample['text'])}")
    print(f"Label type: {type(sample['coarse_label'])}")
    
    if not isinstance(sample['text'], str):
        print("✗ Text is not a string!")
        return False
    
    if not isinstance(sample['coarse_label'], (int, np.integer)):
        print("✗ Label is not an integer!")
        return False
    
    print("✓ Data types are correct")
    print()
    
    # === FINAL VERDICT ===
    print("="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print()
    
    issues = []
    
    if train_size < 1000:
        issues.append("Small training set")
    if imbalance_ratio > 5:
        issues.append("Class imbalance (use class_weights=True)")
    if empty_count > 0:
        issues.append("CRITICAL: Empty texts found!")
    if very_short_count > 10:
        issues.append("Many very short texts")
    
    if issues:
        print("⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print()
        if "CRITICAL" in str(issues):
            print("✗ CRITICAL ISSUES - Must fix before training!")
            return False
        else:
            print("⚠️  Minor issues - Training may work but watch for problems")
    else:
        print("✓ Dataset looks clean!")
    
    print()
    print("RECOMMENDATION:")
    if imbalance_ratio > 5:
        print("  - Use 'use_class_weights': True in config")
    if avg_length > 100:
        print(f"  - Use 'max_seq_length': {int(avg_length * 1.5)} in config")
    
    print("\nExpected performance with good model:")
    print(f"  - Train accuracy: 95-99%")
    print(f"  - Test accuracy: {expected}")
    
    return True


if __name__ == "__main__":
    success = validate_dataset()
    
    if success:
        print("\n" + "="*70)
        print("✓ DATA VALIDATION PASSED")
        print("="*70)
        print("\nYou can proceed with training.")
        print("If training still fails, the problem is with the model/config.")
    else:
        print("\n" + "="*70)
        print("✗ DATA VALIDATION FAILED")
        print("="*70)
        print("\nFix data issues before training!")
    
    sys.exit(0 if success else 1)