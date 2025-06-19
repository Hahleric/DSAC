# ---------- utils.py ----------
import numpy as np, xxhash  # pip install xxhash

def bloom_encode(cache_bits: np.ndarray,
                 B: int = 512,
                 H: int = 4,
                 seed: int = 17) -> np.ndarray:
    """
    把 F 位 bitmap 编成长度 B 的 0/1 向量 (Bloom filter)。
    """
    bloom = np.zeros(B, np.float32)
    idxs  = np.flatnonzero(cache_bits)
    for f in idxs:
        # 多个哈希；xxhash64 比内置 hash 稳定
        for i in range(H):
            h = xxhash.xxh64_intdigest(f"{f}_{i}", seed=seed) % B
            bloom[h] = 1.0
    return bloom
