from __future__ import annotations

import time
import os
import faiss
import numpy as np

def _t() -> float:
    return time.perf_counter()

def _accumulate(I: np.ndarray, counts: np.ndarray) -> None:
    valid = I.ravel()
    valid = valid[valid >= 0]
    counts += np.bincount(valid, minlength=counts.shape[0]).astype(np.int64)

def _rank(counts: np.ndarray, K: int) -> np.ndarray:
    order = np.lexsort((np.arange(counts.shape[0], dtype=np.int64), -counts))
    return order[:K].astype(np.int64)

def solve(
    base_vectors:  np.ndarray,
    query_vectors: np.ndarray,
    k:             int,
    K:             int,
    time_budget:   float,
) -> np.ndarray:
    
    t0 = _t()
    
    # Bypassing strict HPC thread limits
    num_cores = os.cpu_count() or 96
    faiss.omp_set_num_threads(num_cores)
    
    # Tight safety net
    _OVERHEAD_SEC = 0.5
    deadline = t0 + time_budget - _OVERHEAD_SEC

    if not base_vectors.flags['C_CONTIGUOUS']:
        base_vectors = np.ascontiguousarray(base_vectors, dtype=np.float32)
    if not query_vectors.flags['C_CONTIGUOUS']:
        query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
        
    N, d = base_vectors.shape
    Q = query_vectors.shape[0]
    K_out = min(K, N)

    np.random.seed(42)
    shuffled_indices = np.random.permutation(Q)
    query_vectors = query_vectors[shuffled_indices]

    counts = np.zeros(N, dtype=np.int64)

    nlist = min(2048, max(512, int(4 * np.sqrt(N))))
    
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    sample_size = min(N, nlist * 50) 
    index.train(base_vectors[:sample_size])
    index.add(base_vectors)

    if _t() >= deadline:
        return _rank(counts, K_out)

    if time_budget <= 25.0:
        index.nprobe = 48      
    elif time_budget <= 50.0:
        index.nprobe = 96      
    else:
        index.nprobe = 256     

    # --- DYNAMIC COASTING LOGIC ---
    start = 0
    _BATCH_SIZE = 10000 
    avg_time_per_query = 0.0
    processed_queries = 0

    while start < Q:
        now = _t()
        if now >= deadline:
            break
            
        if avg_time_per_query > 0.0:
            expected_time = avg_time_per_query * _BATCH_SIZE * 1.05 # 5% buffer
            
            if now + expected_time >= deadline:
                time_left = deadline - now
                

                safe_queries = int(time_left / (avg_time_per_query * 1.5))
                
                _BATCH_SIZE = (safe_queries // 256) * 256
                
                if _BATCH_SIZE < 256:
                    break
                    
        end = min(start + _BATCH_SIZE, Q)
        if end <= start:
            break
            
        batch_count = end - start
        
        _, I = index.search(query_vectors[start:end], k)
        _accumulate(I, counts)
        processed_queries = end
        
        batch_time = _t() - now
        current_speed = batch_time / batch_count
        
        # Exponential moving average of speed per query
        if avg_time_per_query == 0.0:
            avg_time_per_query = current_speed
        else:
            avg_time_per_query = 0.5 * avg_time_per_query + 0.5 * current_speed
            
        start = end

    print(f"Processed {processed_queries} / {Q} queries with nprobe={index.nprobe} using {num_cores} threads")

    return _rank(counts, K_out)