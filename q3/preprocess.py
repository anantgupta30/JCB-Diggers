import sys
import hashlib

def get_graph_digest(lines):
    """Creates a hash of the graph structure to detect duplicates."""
    # We join lines to create a single string representing the graph
    # This detects exact textual duplicates.
    return hashlib.md5("".join(lines).encode('utf-8')).hexdigest()

def main():
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py <input_graph_file> <output_clean_file>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    unique_hashes = set()
    output_lines = []
    
    current_graph_lines = []
    # Counter for the new, contiguous IDs
    new_graph_id = 0

    # Robust encoding read
    try:
        f = open(input_path, 'r', encoding='utf-8-sig')
        content = f.read()
    except:
        f = open(input_path, 'r', encoding='cp1252')
        content = f.read()
    f.close()

    for line in content.splitlines():
        line = line.strip()
        if not line: continue
        
        if line.startswith('t #'):
            # 1. PROCESS PREVIOUS GRAPH
            if current_graph_lines:
                g_hash = get_graph_digest(current_graph_lines)
                if g_hash not in unique_hashes:
                    unique_hashes.add(g_hash)
                    
                    # RE-INDEXING STEP:
                    # The first line is 't # OLD_ID'. We replace it with 't # NEW_ID'
                    # split indices: 't', '#', 'ID' -> we want 't # NEW_ID'
                    old_header = current_graph_lines[0]
                    # verify it is actually a header
                    if old_header.startswith('t #'):
                         current_graph_lines[0] = f"t # {new_graph_id}"
                    
                    output_lines.extend(current_graph_lines)
                    new_graph_id += 1
            
            # 2. START NEW GRAPH
            current_graph_lines = [line]
        else:
            current_graph_lines.append(line)

    # 3. PROCESS LAST GRAPH
    if current_graph_lines:
        g_hash = get_graph_digest(current_graph_lines)
        if g_hash not in unique_hashes:
            unique_hashes.add(g_hash)
            
            # Re-index last graph
            old_header = current_graph_lines[0]
            if old_header.startswith('t #'):
                 current_graph_lines[0] = f"t # {new_graph_id}"
            
            output_lines.extend(current_graph_lines)
            new_graph_id += 1

    print(f"Processed {len(unique_hashes)} unique graphs (re-indexed 0 to {new_graph_id-1}).")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines) + "\n")

if __name__ == "__main__":
    main()