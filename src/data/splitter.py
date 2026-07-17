import os
import re
import random

# precompile regex to find digit groups
_DIGIT_GROUP_RE = re.compile(r"\d+")

def extract_numeric_index(name):
    """Extract the last contiguous digit group from a filename or path.

    Returns an int index or None if no digit group is found.
    Handles values like '000123.jpg', 'img_000123.png', or full paths.
    """
    if not isinstance(name, str):
        return None

    # strip directory and extension
    base = os.path.basename(name)
    stem, _ext = os.path.splitext(base)

    groups = _DIGIT_GROUP_RE.findall(stem)
    if not groups:
        return None

    # use the last numeric group as the index
    try:
        return int(groups[-1])
    except ValueError:
        return None

def sequence_aware_split(dataset, val_fraction=0.2, gap=5, seed=42):
    numbered = []
    non_numbered = []

    for idx, entry in enumerate(dataset.samples):
        # some dataset implementations store samples as (path, label)
        name = entry[0] if isinstance(entry, (tuple, list)) and entry else entry
        num = extract_numeric_index(name)
        if num is not None:
            numbered.append((num, idx))
        else:
            non_numbered.append(idx)

    # sort numbered by extracted numeric value, keep non-numbered at the end
    numbered.sort(key=lambda x: x[0])
    sorted_indices = [idx for _, idx in numbered] + non_numbered
    n = len(sorted_indices)

    val_size = int(n * val_fraction)
    # ensure at least one validation sample when fraction > 0
    if val_fraction > 0 and val_size == 0 and n > 0:
        val_size = 1

    rng = random.Random(seed)
    latest_start = n - val_size

    val_start = rng.randint(0, max(latest_start, 0))
    val_end = val_start + val_size

    val_idx = sorted_indices[val_start:val_end]
    train_idx = sorted_indices[:max(val_start - gap, 0)] + sorted_indices[min(val_end + gap, n):]

    return train_idx, val_idx