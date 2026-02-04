import sys
import numpy as np

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 generate_candidates.py <db_features> <query_features> <output_file>")
        sys.exit(1)

    db_path = sys.argv[1]
    query_path = sys.argv[2]
    out_path = sys.argv[3]

    m_db = np.load(db_path)
    m_q = np.load(query_path)

    print(f"DB Shape: {m_db.shape}, Query Shape: {m_q.shape}")
    
    all_candidate_counts = []
    with open(out_path, 'w') as f:
        for i_q in range(len(m_q)):
            vec_q = m_q[i_q]
            
            mask = np.all(vec_q <= m_db, axis=1)
            candidate_indices = np.where(mask)[0]
            
            # Serial number of query (0-based)
            f.write(f"q # {i_q}\n")
            c_str = " ".join(map(str, candidate_indices))
            f.write(f"c # {c_str}\n")
            
            all_candidate_counts.append(len(candidate_indices))

    # Print Statistics
    num_features = m_q.shape[1]
    min_cand = np.min(all_candidate_counts)
    max_cand = np.max(all_candidate_counts)
    avg_cand = np.mean(all_candidate_counts)
    
    print("\n" + "="*40)
    print(f"Index Statistics (k={num_features})")
    print(f"Min |C_q| : {min_cand}")
    print(f"Max |C_q| : {max_cand}")
    print(f"Avg |C_q| : {avg_cand:.2f}")
    print("="*40 + "\n")
    print(f"Candidates generated in {out_path}")

if __name__ == "__main__":
    main()
