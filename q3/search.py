import sys
import multiprocessing
import numpy as np

def load_vectors_to_numpy(filename):
    data = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                bits = [int(x) for x in line.split()]
                data.append(bits)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        sys.exit(1)
    return np.array(data, dtype=bool)

def search_worker_numpy(args):
    q_idx, q_vec, db_matrix = args
    # Logic: We want DB rows where (DB & Q) == Q
    # If Q has 1, DB must have 1.
    # Inverse: Fail if (Q=1 and DB=0)
    mismatch = q_vec & (~db_matrix) # 1 only where Q=1 and DB=0
    valid_mask = ~mismatch.any(axis=1) # True where no mismatch
    
    # Get indices (0-based)
    candidates = np.where(valid_mask)[0]
    return (q_idx, candidates)

def main():
    if len(sys.argv) != 4:
        print("Usage: python search.py <db_vectors> <query_vectors> <output_file>")
        sys.exit(1)

    db_vec_file = sys.argv[1]
    query_vec_file = sys.argv[2]
    out_file = sys.argv[3]

    print("Loading Data...")
    db_matrix = load_vectors_to_numpy(db_vec_file)
    query_matrix = load_vectors_to_numpy(query_vec_file)
    
    n_cores = multiprocessing.cpu_count()
    
    tasks = [(i, query_matrix[i], db_matrix) for i in range(len(query_matrix))]

    print(f"Searching {len(tasks)} queries on {n_cores} cores...")
    with multiprocessing.Pool(n_cores) as pool:
        results = pool.map(search_worker_numpy, tasks)
        
    # Sort results by query index just in case
    results.sort(key=lambda x: x[0])

    print(f"Writing results to {out_file}...")
    with open(out_file, 'w') as f:
        for q_idx, candidates in results:
            f.write(f"q # {q_idx}\n")
            cand_str = " ".join(map(str, candidates))
            f.write(f"c # {cand_str}\n")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()