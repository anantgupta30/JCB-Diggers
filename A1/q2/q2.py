import os
import sys
import matplotlib.pyplot as plt

def read_log(log_path):
    gspan_list = []
    fsg_list = []
    gaston_list = []
    support_list = []
        
    data = {'gSpan': {}, 'FSG': {}, 'Gaston': {}}
    
    with open(log_path, "r") as file:
        current_support = None
        for line in file:
            
            if "runtime at" in line and "% support" in line:
                parts = line.split()
                alg = parts[0] 
                
                support_str = parts[3].replace('%', '')
                try:
                    support = int(support_str)
                except ValueError:
                    continue
                    
                time_str = parts[5] 
                try:
                    time_val = int(time_str)
                except ValueError:
                    time_val = 0
                
                if alg in data:
                    data[alg][support] = time_val
                    if support not in support_list:
                        support_list.append(support)

    support_list = sorted(support_list)
    
    gspan_list = [data['gSpan'].get(s, 0) for s in support_list]
    fsg_list = [data['FSG'].get(s, 0) for s in support_list]
    gaston_list = [data['Gaston'].get(s, 0) for s in support_list]
    
    return support_list, gspan_list, fsg_list, gaston_list

def plot_results(support_list, gspan_list, fsg_list, gaston_list, output_path):
    if plt is None:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(support_list, gspan_list, label="gSpan", marker="o")
    plt.plot(support_list, fsg_list, label="FSG", marker="o")
    plt.plot(support_list, gaston_list, label="Gaston", marker="o")
    plt.xlabel("Minimum Support (%)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Comparison of gSpan, FSG, and Gaston")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "plot.png"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 q2.py <output_path>")
        sys.exit(1)
        
    output_path = sys.argv[1]
    log_path = os.path.join(output_path, "run_log.txt")
    
    if os.path.exists(log_path):
        support_list, gspan_list, fsg_list, gaston_list = read_log(log_path)
        plot_results(support_list, gspan_list, fsg_list, gaston_list, output_path)
    else:
        print(f"Log file not found: {log_path}")
