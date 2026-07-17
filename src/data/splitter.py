import re
import random

def extract_numeric_index(name):
    match = re.search(r'(\d+)$', name)

    return int(match.group(1)) if match else None

def sequence_aware_split(dataset, val_fraction=0.2, gap=5, seed=42):
    numbered = []

    for idx, name in enumerate(dataset.samples):
        num = extract_numeric_index(name)
        if num is not None:
            numbered.append((num, idx))

    if len(numbered) != len(dataset.samples):
        raise ValueError("Not all samples have numeric indices in their filenames.")
    
    numbered.sort(key=lambda x: x[0])
    sorted_indices = [idx for _, idx in numbered]
    n = len(sorted_indices)

    val_size = int(n * val_fraction)

    rng = random.Random(seed)
    latest_start = n - val_size

    val_start = rng.randint(0, max(latest_start, 0))
    val_end = val_start + val_size

    val_idx = sorted_indices[val_start:val_end]
    train_idx = sorted_indices[:max(val_start - gap, 0)] + sorted_indices[min(val_end + gap, n):]

    return train_idx, val_idx