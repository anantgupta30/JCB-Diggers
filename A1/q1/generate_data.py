import random
import sys

def generate_dataset(universe_size, num_txns,
                     core_frac=0.10,
                     core_prob=0.90,
                     noise_prob=0.05,
                     outfile="generated_transactions.dat"):
    """
    universe_size : total number of distinct items
    num_txns      : number of transactions

    core_frac  : fraction of items that are very frequent (10%)
    core_prob  : probability core item appears in a transaction (90%)
    noise_prob : probability rare item appears (5%)
    """

    items = list(range(universe_size))

    core_size = max(1, int(core_frac * universe_size))
    core_items = random.sample(items, core_size)

    rare_items = list(set(items) - set(core_items))

    with open(outfile, "w") as f:
        for _ in range(num_txns):
            txn = []

            # frequent core items
            for item in core_items:
                if random.random() < core_prob:
                    txn.append(str(item))

            # rare noisy items
            for item in rare_items:
                if random.random() < noise_prob:
                    txn.append(str(item))

            # ensure non-empty transaction
            if not txn:
                txn.append(str(random.choice(items)))

            random.shuffle(txn)
            f.write(" ".join(txn) + "\n")


if __name__ == "__main__":
    U = int(sys.argv[1])   # universe size
    N = int(sys.argv[2])   # number of transactions

    generate_dataset(U, N)
