# LSH-Fast

A fast and scalable implementation of Locality Sensitive Hashing in Python.

## Installation

```bash
pip install lsh-fast
```

## Usage

```python
from lsh import HashParams, LSHHashFunctions
import numpy as np

# Initialize hash functions
params = HashParams(num_projections=100, projection_dim=1000)
hash_functions = LSHHashFunctions(params)

# Compute hash values for a vector
vector = np.random.randn(1000)
hash_values = hash_functions.compute_hash_values(vector)

# Compute bucket hashes
bucket_hashes = hash_functions.compute_bucket_hash(hash_values, band_size=10)
```

## Features

- Fast and memory-efficient implementation
- Optimized for large-scale data
- Easy to use API
- Configurable parameters

## License

MIT License