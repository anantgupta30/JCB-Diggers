import pandas as pd
import matplotlib.pyplot as plt
import sys

# Load runtime data
def plot_graph(csv_file,out_file):
    df = pd.read_csv(csv_file)

    # Sort by increasing support
    df = df.sort_values("support")
    plt.figure(figsize=(8,5))
    print(df['apriori_time'])
    print(df['fpgrowth_time'])
    plt.plot(df["support"], df["apriori_time"], marker='o', label="Apriori")
    plt.plot(df["support"], df["fpgrowth_time"], marker='o', label="FP-growth")
    # plt.xticks(range(0, 101, 5))
    plt.xlabel("Support Threshold (%)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Apriori vs FP-growth Runtime Comparison")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"output/{out_file}")

if __name__ == "__main__":
    csv_file = sys.argv[1]
    out_file = sys.argv[2]
    plot_graph(csv_file, out_file)