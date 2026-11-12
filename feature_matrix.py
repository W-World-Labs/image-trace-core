"""Pure NumPy similarity matrices for caller-supplied vectors (MIT)."""
from typing import Dict, List

import numpy as np

NUM_VARIANTS = 8

def hash_str_to_bits(hash_str: str) -> np.ndarray:
    """Convert hex hash string (e.g. 'a3f1...') to 64-dim uint8 bit vector."""
    if not hash_str:
        return np.zeros(64, dtype=np.uint8)
    # Each hex char = 4 bits
    bits = []
    for ch in hash_str[:16]:
        val = int(ch, 16)
        bits.extend([(val >> (3 - b)) & 1 for b in range(4)])
    arr = np.array(bits[:64], dtype=np.uint8)
    if len(arr) < 64:
        arr = np.pad(arr, (0, 64 - len(arr)))
    return arr

def _build_matrix_from_vectors(
    vectors: Dict[int, Dict[int, np.ndarray]],
    image_ids: List[int],
    variant: int,
    dtype=np.float64,
) -> np.ndarray:
    """Stack complete, same-sized vectors for a given variant."""
    n = len(image_ids)
    sample = None
    for d in vectors.values():
        if variant in d:
            sample = d[variant]
            break
    if sample is None:
        raise ValueError(f"No feature vectors available for variant {variant}")
    dim = len(sample)
    mat = np.zeros((n, dim), dtype=dtype)
    for i, img_id in enumerate(image_ids):
        vector = vectors.get(img_id, {}).get(variant)
        if vector is None:
            raise ValueError(f"Missing feature vector for image {img_id}, variant {variant}")
        if len(vector) != dim:
            raise ValueError(
                f"Feature vector dimension mismatch for image {img_id}, variant {variant}: "
                f"expected {dim}, got {len(vector)}"
            )
        mat[i] = vector.astype(dtype)
    return mat

def compute_hash_similarity_matrix(
    vectors: Dict[int, Dict[int, np.ndarray]],
    image_ids: List[int],
    rotation_invariant: bool = False,
) -> np.ndarray:
    """
    Compute N×N hash similarity using batch XOR + popcount.

    Non-rotation: single numpy broadcast → microseconds.
    Rotation-invariant: max over variant matrices → still fast.
    """
    n = len(image_ids)
    if n == 0:
        return np.array([], dtype=np.float64).reshape(0, 0)

    variants = list(range(NUM_VARIANTS)) if rotation_invariant else [0]
    best_sim = np.zeros((n, n), dtype=np.float64)

    # Compare each orientation of A against each orientation of B. Comparing
    # only matching variant indexes fails when the two source images differ by
    # a relative rotation (e.g. original-vs-90-degree-rotated).
    for variant_a in variants:
        mat_a = _build_matrix_from_vectors(vectors, image_ids, variant_a, dtype=np.uint8)
        for variant_b in variants:
            mat_b = _build_matrix_from_vectors(vectors, image_ids, variant_b, dtype=np.uint8)
            if mat_a.shape[1] != mat_b.shape[1]:
                raise ValueError("Hash feature dimensions differ between variants")
            bits = mat_a.shape[1]
            xor = mat_a[:, None, :] ^ mat_b[None, :, :]
            hamming = xor.sum(axis=2).astype(np.float64)
            sim = 1.0 - hamming / max(bits, 1)
            np.maximum(best_sim, sim, out=best_sim)

    np.fill_diagonal(best_sim, 1.0)
    return best_sim

def compute_cosine_similarity_matrix(
    vectors: Dict[int, Dict[int, np.ndarray]],
    image_ids: List[int],
    rotation_invariant: bool = False,
) -> np.ndarray:
    """
    Compute N×N cosine similarity using batch matrix multiplication (BLAS).

    Pure numpy mat @ mat.T — already sub-millisecond for 100 images.
    """
    n = len(image_ids)
    if n == 0:
        return np.array([], dtype=np.float64).reshape(0, 0)

    variants = list(range(NUM_VARIANTS)) if rotation_invariant else [0]
    best_sim = np.zeros((n, n), dtype=np.float64)

    for variant_a in variants:
        mat_a = _build_matrix_from_vectors(vectors, image_ids, variant_a, dtype=np.float64)
        for variant_b in variants:
            mat_b = _build_matrix_from_vectors(vectors, image_ids, variant_b, dtype=np.float64)
            if mat_a.shape[1] != mat_b.shape[1]:
                raise ValueError("Feature dimensions differ between variants")
            norms_a = np.linalg.norm(mat_a, axis=1, keepdims=True)
            norms_b = np.linalg.norm(mat_b, axis=1, keepdims=True)
            normed_a = mat_a / np.where(norms_a > 0, norms_a, 1.0)
            normed_b = mat_b / np.where(norms_b > 0, norms_b, 1.0)
            sim = np.clip(normed_a @ normed_b.T, 0.0, 1.0)
            np.maximum(best_sim, sim, out=best_sim)

    np.fill_diagonal(best_sim, 1.0)
    return best_sim
