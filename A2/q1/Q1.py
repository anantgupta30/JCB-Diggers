import sys
import urllib.request
import json
import numpy as np
import matplotlib.pyplot as plt

def get_data(dataset_num):
    url = f"http://hulk.cse.iitd.ac.in:3000/dataset?student_id=aib252556&dataset_num={dataset_num}"
    with urllib.request.urlopen(url) as response:
        raw_data = response.read().decode('utf-8')
        data = json.loads(raw_data)
    return np.array(data["X"])

def kmeans_plusplus(X, k):
    n_samples, n_features = X.shape
    centroids = np.empty((k, n_features))
    
    idx = np.random.randint(n_samples)
    centroids[0] = X[idx]
    
    for i in range(1, k):
        distances = np.min(np.sum((X[:, np.newaxis] - centroids[:i])**2, axis=2), axis=1)
        
        total_dist = np.sum(distances)
        if total_dist == 0:
            idx = np.random.randint(n_samples)
        else:
            probs = distances / total_dist
            idx = np.random.choice(n_samples, p=probs)
        centroids[i] = X[idx]
        
    return centroids

def kmeans(X, k, max_iters=100, n_init=10):
    best_centroids = None
    best_sse = np.inf
    
    for _ in range(n_init):
        centroids = kmeans_plusplus(X, k)
        
        for i in range(max_iters):
            distances = np.sum((X[:, np.newaxis] - centroids)**2, axis=2)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j] for j in range(k)])
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
            
        final_distances = np.sum((X[:, np.newaxis] - centroids)**2, axis=2)
        sse = np.sum(np.min(final_distances, axis=1))
        
        if sse < best_sse:
            best_sse = sse
            best_centroids = centroids
            
    return best_centroids, best_sse

def solve():
    if len(sys.argv) < 2:
        print("Usage: python3 Q1.py <dataset_num> OR python3 Q1.py <path_to_dataset>.npy")
        return

    arg = sys.argv[1]
    if arg.isdigit():
        data = get_data(1)
        data2 = get_data(2)
        sse_values = []
        ks = list(range(1, 16))
        sse_values_2 = []
        for k in ks:
            _, sse = kmeans(data, k)
            sse_values.append(sse)
            _, sse2 = kmeans(data2, k)
            sse_values_2.append(sse2)

        p1 = np.array([ks[0], sse_values[0]])
        pn = np.array([ks[-1], sse_values[-1]])
        
        distances = []
        for i in range(len(ks)):
            p = np.array([ks[i], sse_values[i]])
            # Distance from point p to line segment p1-pn
            d = np.abs(np.cross(pn - p1, p1 - p)) / np.linalg.norm(pn - p1)
            distances.append(d)
            
        optimal_k = ks[np.argmax(distances)]
        print("Optimal for dataset 1:",optimal_k)

        for i in range(len(ks)):
            p = np.array([ks[i], sse_values_2[i]])
            # Distance from point p to line segment p1-pn
            d = np.abs(np.cross(pn - p1, p1 - p)) / np.linalg.norm(pn - p1)
            distances.append(d)
            
        optimal_k_2 = ks[np.argmax(distances)]
        print("Optimal for dataset 2:",optimal_k_2)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ---- Plot 1 ----
        axes[0].plot(ks, sse_values, marker='o', linestyle='-', color='b')
        axes[0].axvline(x=optimal_k, linestyle=':', color='black', label=f'Optimal k = {optimal_k}')
        axes[0].set_xlabel('Number of clusters (k)')
        axes[0].set_ylabel('Objective value (SSE)')
        axes[0].set_title('Dataset 1: k-means Objective vs k')
        axes[0].grid(True)
        axes[0].legend()
        axes[0].set_title("Mode 1")

        # ---- Plot 2 ----
        axes[1].plot(ks, sse_values_2, marker='s', linestyle='--', color='r')
        axes[1].axvline(x=optimal_k_2, linestyle=':', color='black', label=f'Optimal k = {optimal_k_2}')
        axes[1].set_xlabel('Number of clusters (k)')
        axes[1].set_ylabel('Objective value (SSE)')
        axes[1].set_title('Dataset 2: k-means Objective vs k')
        axes[1].grid(True)
        axes[1].legend()
        axes[1].set_title("Mode 2")

        plt.tight_layout()
        plt.grid(True)
        plt.savefig('plot.png')
    elif arg.endswith('.npy'):
        try:
            data_obj = np.load(arg, allow_pickle=True)
            if isinstance(data_obj, np.ndarray):
                data = data_obj
            else:
                data = np.array(data_obj.item()["X"]) if hasattr(data_obj, 'item') else np.array(data_obj["X"])
        except Exception:
            with open(arg, 'r') as f:
                data_json = json.load(f)
            data = np.array(data_json["X"])
            sse_values = []
            ks = list(range(1, 16))
            
            for k in ks:
                _, sse = kmeans(data, k)
                sse_values.append(sse)

            p1 = np.array([ks[0], sse_values[0]])
            pn = np.array([ks[-1], sse_values[-1]])
            
            distances = []
            for i in range(len(ks)):
                p = np.array([ks[i], sse_values[i]])
                # Distance from point p to line segment p1-pn
                d = np.abs(np.cross(pn - p1, p1 - p)) / np.linalg.norm(pn - p1)
                distances.append(d)
                
            optimal_k = ks[np.argmax(distances)]
            print(optimal_k)
            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(ks, sse_values, marker='o', linestyle='-', color='b')
            plt.axvline(x=optimal_k, linestyle=':', color='black', label=f'Optimal k = {optimal_k}')
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Objective value (SSE)')
            plt.title('k-means Objective vs k')
            plt.grid(True)
            plt.legend()
            plt.savefig('plot.png')
            
            # Optimal k using elbow method (geometric approach)
            # Line from (k=1, SSE_1) to (k=15, SSE_15)

    else:
        print("Invalid argument format.")
        return



    

if __name__ == "__main__":
    solve()
