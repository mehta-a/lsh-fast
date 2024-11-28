import numpy as np
from typing import Union, List, Tuple, Optional
from numpy.typing import NDArray
import xxhash
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

@dataclass
class HashParams:
    """Parameters for hash functions"""
    num_projections: int
    projection_dim: int
    seed: int = 42
    chunk_size: int = 10000
    n_jobs: int = multiprocessing.cpu_count()

class LSHHashFunctions:
    """Optimized implementation of hash functions for LSH"""
    
    def __init__(self, params: HashParams):
        """Initialize hash function parameters and random projections"""
        self.params = params
        np.random.seed(params.seed)
        
        # Pre-compute random projections matrix for better performance
        self._random_projections = np.random.normal(
            size=(params.num_projections, params.projection_dim)
        ).astype(np.float32)
        
        # Initialize xxhash objects for each projection
        self._xxhash_objects = [
            xxhash.xxh64(seed=params.seed + i) 
            for i in range(params.num_projections)
        ]

    def compute_hash_values(
        self, 
        input_vector: Union[NDArray, List[float]], 
        num_hashes: Optional[int] = None
    ) -> NDArray:
        """
        Compute LSH hash values for input vector using random projections
        
        Args:
            input_vector: Input vector to hash
            num_hashes: Optional number of hash values to compute (default: all)
            
        Returns:
            Array of hash values
        """
        if not isinstance(input_vector, np.ndarray):
            input_vector = np.array(input_vector, dtype=np.float32)
            
        # Ensure input vector has correct shape
        if input_vector.shape[-1] != self.params.projection_dim:
            raise ValueError(
                f"Input vector dimension {input_vector.shape[-1]} does not match "
                f"projection dimension {self.params.projection_dim}"
            )
            
        # Calculate number of hashes to compute
        n_hashes = num_hashes or self.params.num_projections
        
        # Compute dot products efficiently using matrix multiplication
        projections = np.dot(
            self._random_projections[:n_hashes], 
            input_vector
        )
        
        # Apply sign function to get binary hash values
        return np.sign(projections)
    
    def compute_bucket_hash(
        self, 
        hash_values: NDArray, 
        band_size: int
    ) -> List[int]:
        """
        Compute bucket hashes for LSH using xxhash for better performance
        
        Args:
            hash_values: Array of hash values from compute_hash_values
            band_size: Size of each band for bucket hashing
            
        Returns:
            List of bucket hash values
        """
        num_bands = len(hash_values) // band_size
        bucket_hashes = []
        
        for i in range(num_bands):
            band = hash_values[i * band_size : (i + 1) * band_size]
            
            # Reset xxhash object
            xxhash_obj = self._xxhash_objects[i]
            xxhash_obj.reset()
            
            # Update with band values
            xxhash_obj.update(band.tobytes())
            
            # Get hash value
            bucket_hashes.append(xxhash_obj.intdigest())
            
        return bucket_hashes
