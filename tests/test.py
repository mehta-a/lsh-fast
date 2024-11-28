# File: tests/test_hash_functions.py
import numpy as np
import pytest
from lsh import HashParams, LSHHashFunctions

def test_hash_functions_initialization():
    params = HashParams(num_projections=100, projection_dim=1000)
    hash_functions = LSHHashFunctions(params)
    assert hash_functions._random_projections.shape == (100, 1000)

def test_compute_hash_values():
    params = HashParams(num_projections=100, projection_dim=10)
    hash_functions = LSHHashFunctions(params)
    vector = np.random.randn(10)
    hash_values = hash_functions.compute_hash_values(vector)
    assert hash_values.shape == (100,)
    assert np.all(np.abs(hash_values) == 1)

def test_compute_bucket_hash():
    params = HashParams(num_projections=100, projection_dim=10)
    hash_functions = LSHHashFunctions(params)
    vector = np.random.randn(10)
    hash_values = hash_functions.compute_hash_values(vector)
    bucket_hashes = hash_functions.compute_bucket_hash(hash_values, band_size=10)
    assert len(bucket_hashes) == 10