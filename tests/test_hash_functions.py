import numpy as np
import pytest
from lsh import HashParams, LSHHashFunctions

def test_hash_params_initialization():
    """Test if HashParams can be initialized with correct values"""
    params = HashParams(num_projections=100, projection_dim=1000)
    assert params.num_projections == 100
    assert params.projection_dim == 1000
    assert params.seed == 42  # default value

def test_lsh_initialization():
    """Test if LSHHashFunctions initializes correctly"""
    params = HashParams(num_projections=100, projection_dim=1000)
    lsh = LSHHashFunctions(params)
    assert lsh._random_projections.shape == (100, 1000)
    assert len(lsh._xxhash_objects) == 100

def test_compute_hash_values():
    """Test if hash values are computed correctly"""
    params = HashParams(num_projections=100, projection_dim=10)
    lsh = LSHHashFunctions(params)
    
    # Test single vector
    vector = np.random.randn(10)
    hash_values = lsh.compute_hash_values(vector)
    assert hash_values.shape == (100,)
    assert np.all(np.abs(hash_values) == 1)  # Should only contain 1 or -1

def test_compute_bucket_hash():
    """Test if bucket hashes are computed correctly"""
    params = HashParams(num_projections=100, projection_dim=10)
    lsh = LSHHashFunctions(params)
    
    # Generate hash values and compute bucket hashes
    vector = np.random.randn(10)
    hash_values = lsh.compute_hash_values(vector)
    bucket_hashes = lsh.compute_bucket_hash(hash_values, band_size=10)
    
    assert len(bucket_hashes) == 10  # Should have num_projections/band_size buckets
    assert all(isinstance(x, int) for x in bucket_hashes)  # All bucket hashes should be integers

def test_invalid_input_dimension():
    """Test if proper error is raised for invalid input dimensions"""
    params = HashParams(num_projections=100, projection_dim=10)
    lsh = LSHHashFunctions(params)
    
    # Test vector with wrong dimension
    wrong_dim_vector = np.random.randn(20)  # dimension should be 10
    with pytest.raises(ValueError):
        lsh.compute_hash_values(wrong_dim_vector)

def test_list_input():
    """Test if the function works with list inputs"""
    params = HashParams(num_projections=100, projection_dim=10)
    lsh = LSHHashFunctions(params)
    
    # Test with list input
    vector_list = [1.0] * 10
    hash_values = lsh.compute_hash_values(vector_list)
    assert hash_values.shape == (100,)