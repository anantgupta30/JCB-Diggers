import argparse
import gc
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

from load_dataset import load_dataset

# Models and utilities for A
from model_a import GCN_A, GraphSAGE_A, NodeClassifierA
from model_a import iter_batches as iter_batches_a

# Models and utilities for B
from model_b import FeatureMLPWithPropagation, make_pos_weight
from model_b import iter_batches as iter_batches_b

# Models and utilities for C
from model_c import DualSignalLinkPredictorC, sample_negative_edges


# =============================================================================
# DATASET A: HELPER FUNCTIONS & TRAINING LOOP
# =============================================================================

@torch.no_grad()
def estimate_normalizer_a(x, node_idx, max_nodes):
    if node_idx.numel() > max_nodes:
        node_idx = node_idx[torch.randperm(node_idx.numel())[:max_nodes]]
    sample = torch.nan_to_num(x[node_idx].float(), nan=0.0, posinf=0.0, neginf=0.0)
    return sample.mean(dim=0).cpu(), sample.std(dim=0).clamp_min(1e-6).cpu()

@torch.no_grad()
def evaluate_a(model, data, val_nodes, device):
    if hasattr(model, "predict_all"):
        pred = model.predict_all(data, device=device)
    else:
        model.eval()
        logits = model(data.x.to(device), data.edge_index.to(device)).cpu()
        pred = logits.argmax(dim=1)
    y_true = data.y[data.val_mask].cpu().numpy()
    y_pred = pred[val_nodes].cpu().numpy()
    return float(accuracy_score(y_true, y_pred))

@torch.no_grad()
def tune_propagation_a(model, data, val_nodes, alphas, steps_list, device):
    logits = model.predict_logits(data.x, device=device)
    probs = torch.softmax(logits, dim=1)
    y_true = data.y[data.val_mask].cpu().numpy()
    best_acc, best_alpha, best_steps = -1.0, 0.0, 0
    for steps in steps_list:
        for alpha in alphas:
            model.propagation_steps = int(steps)
            model.propagation_alpha = float(alpha)
            smoothed = model.propagate(probs, data.edge_index)
            acc = float(accuracy_score(y_true, smoothed[val_nodes].argmax(dim=1).numpy()))
            print(f"propagation steps={steps} alpha={alpha:.2f} val_acc={acc:.5f}")
            if acc > best_acc:
                best_acc, best_alpha, best_steps = acc, float(alpha), int(steps)
            del smoothed
            gc.collect()
    model.propagation_steps = best_steps
    model.propagation_alpha = best_alpha
    return best_acc, best_alpha, best_steps

def run_train_a(args):
    epochs = 80
    batch_size = 4096
    eval_batch_size = 65536
    hidden_channels = 256
    num_layers = 3
    dropout = 0.35
    lr = 2e-3
    weight_decay = 1e-4
    model_type = "gcn"  
    patience = 80
    normalizer_nodes = 200000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Dataset {args.dataset} from {args.data_dir} via load_dataset.py...")
    dataset = load_dataset(args.dataset, args.data_dir)
    data = dataset[0]
    
    data.edge_index = data.edge_index.long()
    data.y = data.y.long()
    data.labeled_nodes = data.labeled_nodes.long()
    data.train_mask = data.train_mask.bool()
    data.val_mask = data.val_mask.bool()
    data.y_full = torch.full((data.num_nodes,), -1, dtype=torch.long)
    data.y_full[data.labeled_nodes] = data.y

    train_nodes = data.labeled_nodes[data.train_mask]
    val_nodes = data.labeled_nodes[data.val_mask]
    num_classes = int(data.y.max().item()) + 1
    print(data)
    print(f"Train nodes={train_nodes.numel():,} Val nodes={val_nodes.numel():,} Classes={num_classes} Device={device}")

    if model_type == "gcn":
        model = GCN_A(data.x.size(-1), hidden_channels, num_classes, dropout=dropout)
    elif model_type == "sage":
        model = GraphSAGE_A(data.x.size(-1), hidden_channels, num_classes, dropout=dropout)
    else:
        mean, std = estimate_normalizer_a(data.x, train_nodes, normalizer_nodes)
        model = NodeClassifierA(
            in_channels=data.x.size(-1),
            num_classes=num_classes,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            batch_size=eval_batch_size,
        )
        model.set_normalizer(mean, std)
        model.set_label_anchors(train_nodes, data.y[data.train_mask])
    
    model.to(device)
    print(f"Model={model_type} Params={sum(p.numel() for p in model.parameters()):,}")

    class_count = torch.bincount(data.y[data.train_mask], minlength=num_classes).float()
    class_weight = class_count.sum() / class_count.clamp_min(1.0)
    class_weight = class_weight / class_weight.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_acc = -1.0
    patience_count = 0
    best_path = os.path.join(args.model_dir, f"{args.kerberos}_model_{args.dataset}.pt")
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        
        if model_type in ("gcn", "sage"):
            x = data.x.to(device, non_blocking=True)
            edge_index = data.edge_index.to(device, non_blocking=True)
            y = data.y[data.train_mask].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, edge_index)
            loss = criterion(logits[train_nodes.to(device)], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            del x, edge_index, y
        else:
            pbar = tqdm(iter_batches_a(train_nodes, batch_size, shuffle=True), desc=f"epoch {epoch}/{epochs}", leave=False)
            for batch_nodes in pbar:
                xb = data.x[batch_nodes].to(device, non_blocking=True)
                yb = data.y_full[batch_nodes].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                pbar.set_postfix(loss=sum(losses[-20:]) / max(1, len(losses[-20:])))
                del xb, yb
                
        scheduler.step()

        model.propagation_steps = 0
        model.propagation_alpha = 0.0
        acc = evaluate_a(model, data, val_nodes, device)
        print(f"Epoch {epoch:03d} loss={sum(losses)/max(1, len(losses)):.4f} raw_val_acc={acc:.5f} time={time.time()-t0:.1f}s")
        
        if acc > best_acc:
            best_acc = acc
            patience_count = 0
            torch.save(model.cpu(), best_path)
            model.to(device)
            print(f"Saved raw best model to {best_path}")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stop at epoch {epoch}; best raw val accuracy={best_acc:.5f}")
                break
                
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Best raw val accuracy: {best_acc:.5f}")
    best_model = torch.load(best_path, weights_only=False, map_location="cpu")
    
    if hasattr(best_model, "predict_logits"):
        best_acc, best_alpha, best_steps = tune_propagation_a(
            best_model,
            data,
            val_nodes,
            alphas=(0.10, 0.20, 0.35, 0.50, 0.65, 0.80),
            steps_list=(0, 1, 2, 3, 5, 8),
            device=device,
        )
    else:
        best_alpha, best_steps = 0.0, 0
        
    torch.save(best_model.cpu(), best_path)
    print(f"Saved final model to {best_path}")
    print(f"Best final val accuracy: {best_acc:.5f} with steps={best_steps}, alpha={best_alpha:.2f}")


# =============================================================================
# DATASET B: HELPER FUNCTIONS & TRAINING LOOP
# =============================================================================

@torch.no_grad()
def estimate_normalizer_b(x: torch.Tensor, node_idx: torch.Tensor, max_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    if node_idx.numel() > max_nodes:
        node_idx = node_idx[torch.randperm(node_idx.numel())[:max_nodes]]
    sample = x[node_idx].float()
    sample = torch.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)
    mean = sample.mean(dim=0)
    std = sample.std(dim=0).clamp_min(1e-6)
    del sample
    return mean.cpu(), std.cpu()

@torch.no_grad()
def predict_nodes_b(model, x, nodes, device, batch_size):
    model.eval()
    out = torch.empty(nodes.numel(), dtype=torch.float32)
    for out_idx in iter_batches_b(torch.arange(nodes.numel()), batch_size, shuffle=False):
        n = nodes[out_idx]
        xb = x[n].to(device, non_blocking=True)
        out[out_idx] = torch.sigmoid(model(xb)).view(-1).cpu()
        del xb
    return out

def evaluate_raw_auc_b(model, data, val_nodes, device, batch_size):
    y_score = predict_nodes_b(model, data.x, val_nodes, device, batch_size).numpy()
    y_true = data.y[data.val_mask].cpu().numpy()
    return float(roc_auc_score(y_true, y_score))

@torch.no_grad()
def tune_propagation_b(model, data, val_nodes, raw_scores, alphas, steps_list):
    best_auc = -1.0
    best_alpha = 0.0
    best_steps = 0
    y_true = data.y[data.val_mask].cpu().numpy()

    for steps in steps_list:
        for alpha in alphas:
            model.propagation_steps = int(steps)
            model.propagation_alpha = float(alpha)
            smoothed = model.propagate(raw_scores, data.edge_index)
            auc = float(roc_auc_score(y_true, smoothed[val_nodes].numpy()))
            print(f"propagation steps={steps} alpha={alpha:.2f} val_auc={auc:.5f}")
            if auc > best_auc:
                best_auc = auc
                best_alpha = float(alpha)
                best_steps = int(steps)
            del smoothed
            gc.collect()

    model.propagation_alpha = best_alpha
    model.propagation_steps = best_steps
    return best_auc, best_alpha, best_steps

def run_train_b(args):
    epochs = 12
    batch_size = 8192
    eval_batch_size = 65536
    hidden_channels = 512
    num_layers = 3
    dropout = 0.25
    lr = 2e-3
    weight_decay = 1e-4
    normalizer_nodes = 200000
    propagation_chunk_size = 5000000
    amp = True  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    print(f"Loading Dataset {args.dataset} from {args.data_dir} via load_dataset.py...")
    dataset = load_dataset(args.dataset, args.data_dir)
    data = dataset[0]  
    
    data.edge_index = data.edge_index.long()
    data.y = data.y.long()
    data.labeled_nodes = data.labeled_nodes.long()
    data.train_mask = data.train_mask.bool()
    data.val_mask = data.val_mask.bool()
    data.y_full = torch.full((data.num_nodes,), -1, dtype=torch.int8)
    data.y_full[data.labeled_nodes] = data.y.to(torch.int8)

    train_nodes = data.labeled_nodes[data.train_mask]
    val_nodes = data.labeled_nodes[data.val_mask]
    y_train = data.y[data.train_mask].float()
    print(data)
    print(f"Train nodes={train_nodes.numel():,} Val nodes={val_nodes.numel():,} Device={device}")

    mean, std = estimate_normalizer_b(data.x, train_nodes, normalizer_nodes)
    model = FeatureMLPWithPropagation(
        in_channels=data.x.size(-1),
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        batch_size=eval_batch_size,
        propagation_chunk_size=propagation_chunk_size,
    )
    model.set_normalizer(mean, std)
    model.set_label_anchors(train_nodes, y_train)
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=make_pos_weight(y_train).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    best_raw_auc = -1.0
    best_path = os.path.join(args.model_dir, f"{args.kerberos}_model_{args.dataset}.pt")
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []

        pbar = tqdm(iter_batches_b(train_nodes, batch_size, shuffle=True), desc=f"epoch {epoch}/{epochs}", leave=False)
        for batch_nodes in pbar:
            xb = data.x[batch_nodes].to(device, non_blocking=True)
            yb = data.y_full[batch_nodes].float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                logits = model(xb).view(-1)
                loss = criterion(logits, yb)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=sum(losses[-20:]) / max(1, len(losses[-20:])))
            del xb, yb

        scheduler.step()
        raw_auc = evaluate_raw_auc_b(model, data, val_nodes, device, eval_batch_size)
        avg_loss = sum(losses) / max(1, len(losses))
        print(f"Epoch {epoch:02d} loss={avg_loss:.4f} raw_val_auc={raw_auc:.5f} time={time.time()-t0:.1f}s")

        if raw_auc > best_raw_auc:
            best_raw_auc = raw_auc
            model.propagation_steps = 0
            model.propagation_alpha = 0.0
            torch.save(model.cpu(), best_path)
            model.to(device)
            print(f"Saved raw best model to {best_path}")

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print(f"Best raw val AUC: {best_raw_auc:.5f}")

    print("Loading best raw model and tuning graph propagation...")
    best_model = torch.load(best_path, weights_only=False, map_location="cpu")
    raw_scores = best_model.predict_mlp(data.x, device=device, batch_size=eval_batch_size)
    best_auc, best_alpha, best_steps = tune_propagation_b(
        best_model,
        data,
        val_nodes,
        raw_scores,
        alphas=(0.15, 0.30, 0.45, 0.60, 0.75),
        steps_list=(0, 1, 2, 3),
    )
    torch.save(best_model.cpu(), best_path)
    print(f"Saved final model to {best_path}")
    print(f"Best final val AUC: {best_auc:.5f} with steps={best_steps}, alpha={best_alpha:.2f}")


# =============================================================================
# DATASET C: HELPER FUNCTIONS & TRAINING LOOP
# =============================================================================

def hits_at_k(pos_scores: torch.Tensor, neg_scores: torch.Tensor, k: int = 50) -> float:
    n_neg_higher = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
    return (n_neg_higher < k).float().mean().item()

@torch.no_grad()
def evaluate_hits_c(model, data, device, score_batch_size):
    pos_scores = model.score_edges(data.x, data.edge_index, data.valid_pos, device=device, batch_size=score_batch_size)
    p, k, _ = data.valid_neg.shape
    neg_scores = model.score_edges(
        data.x,
        data.edge_index,
        data.valid_neg.view(p * k, 2),
        device=device,
        batch_size=score_batch_size,
    ).view(p, k)
    return hits_at_k(pos_scores, neg_scores, k=50)

def run_train_c(args):
    epochs = 200
    score_batch_size = 262144
    hidden_channels = 256
    embed_channels = 128
    svd_dim = 128
    dropout = 0.30
    lr = 1e-3
    eval_every = 5
    patience = 50
    neg_source = "random" 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Dataset {args.dataset} from {args.data_dir} via load_dataset.py...")
    data = load_dataset(args.dataset, args.data_dir) 
    
    print(data)
    print(f"x={tuple(data.x.shape)} train_pos={tuple(data.train_pos.shape)} train_neg={tuple(data.train_neg.shape) if hasattr(data, 'train_neg') else 'None'} device={device}")

    print(f"Fitting TruncatedSVD with n_components={svd_dim}")
    x_np = data.x.detach().cpu().numpy().astype(np.float32, copy=False)
    svd = TruncatedSVD(n_components=svd_dim, n_iter=7, random_state=42)
    normalize(svd.fit_transform(x_np), norm="l2", axis=1)
    print(f"SVD explained variance ratio sum={svd.explained_variance_ratio_.sum():.4f}")
    del x_np
    gc.collect()

    model = DualSignalLinkPredictorC(
        raw_in_channels=data.x.size(-1),
        in_channels=128, 
        hidden_channels=hidden_channels,
        embed_channels=embed_channels,
        dropout=0.40, 
        score_batch_size=score_batch_size,
    )
    model.to(device)

    x_device = data.x.to(device, non_blocking=True)
    edge_index_device = data.edge_index.to(device, non_blocking=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    existing_edges = set(map(tuple, data.train_pos.cpu().tolist()))
    pos_train = data.train_pos.cpu().long()
    provided_neg = data.train_neg.cpu().long() if hasattr(data, "train_neg") else None

    best_hits = -1.0
    patience_count = 0
    best_path = os.path.join(args.model_dir, f"{args.kerberos}_model_{args.dataset}.pt")
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        
        use_provided = provided_neg is not None and neg_source in ("provided", "mixed") and (
            neg_source == "provided" or epoch % 2 == 1
        )
        
        if use_provided:
            n_pair = min(pos_train.size(0), provided_neg.size(0))
            perm = torch.randperm(n_pair)
            pos_epoch = pos_train[:n_pair][perm]
            neg_train = provided_neg[:n_pair][perm]
        else:
            pos_epoch = pos_train
            neg_train = sample_negative_edges(pos_epoch, data.x.size(0), existing_edges)
            
        optimizer.zero_grad(set_to_none=True)
        loss = model.bpr_loss(
            x_device,
            edge_index_device,
            pos_epoch.to(device, non_blocking=True),
            neg_train.to(device, non_blocking=True),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            hits = evaluate_hits_c(model, data, device, score_batch_size)
            print(f"Epoch {epoch:03d} bpr_loss={float(loss.detach().cpu()):.4f} val_hits50={hits:.5f} time={time.time()-t0:.1f}s")
            
            if hits > best_hits:
                best_hits = hits
                patience_count = 0
                torch.save(model.cpu(), best_path)
                model.to(device)
                print(f"Saved best model to {best_path}")
            else:
                patience_count += 1
                if patience_count >= patience:
                    print(f"Early stop at epoch {epoch}; best val Hits@50={best_hits:.5f}")
                    break
        else:
            print(f"Epoch {epoch:03d} bpr_loss={float(loss.detach().cpu()):.4f} time={time.time()-t0:.1f}s")

    print(f"Best val Hits@50: {best_hits:.5f}")


# =============================================================================
# MAIN DISPATCHER
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Train Script for Datasets A, B, and C.")
    
    # Strictly REQUIRED arguments per your command structure
    parser.add_argument("--dataset", required=True, choices=["A", "B", "C"], help="Dataset to load")
    parser.add_argument("--task", required=True, choices=["node", "link"], help="Task type")
    parser.add_argument("--data_dir", required=True, help="Absolute path to datasets")
    parser.add_argument("--model_dir", required=True, help="Path to save models")
    parser.add_argument("--kerberos", required=True, help="Your kerberos ID")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    # Route execution based on inputs
    if args.dataset == "A":
        if args.task != "node":
            raise ValueError(f"Dataset A is a Node Classification task. Expected --task node, got {args.task}")
        run_train_a(args)
        
    elif args.dataset == "B":
        if args.task != "node":
            raise ValueError(f"Dataset B is a Node Classification task. Expected --task node, got {args.task}")
        run_train_b(args)
        
    elif args.dataset == "C":
        if args.task != "link":
            raise ValueError(f"Dataset C is a Link Prediction task. Expected --task link, got {args.task}")
        run_train_c(args)


if __name__ == "__main__":
    main()