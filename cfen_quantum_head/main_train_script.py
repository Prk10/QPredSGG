import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np 
import h5py
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import load_config, update_config, parse_args
from utils.seed import set_seed
from utils.metrics import recall_at_k, mean_recall_at_k
from data.sgg_dataset import NPZSceneGraphDataset, SyntheticSGGDataset, collate_fn
from models.cfen import CFEN


def get_class_weights(h5_path, device):
    print(f"Calculating class weights from {h5_path}...")
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'predicates' in f:
                all_rels = f['predicates'][:].flatten()
            else:
                print("'predicates' key not found in H5. Using uniform weights.")
                return None
    except Exception as e:
        print(f"Error loading H5 for weights: {e}. Using uniform weights.")
        return None

    counts = np.bincount(all_rels, minlength=51)
    counts = np.maximum(counts, 1)
    
    n_samples = len(all_rels)
    n_classes = len(counts)
    weights = n_samples / (n_classes * counts)
    
    weights = weights / weights.mean()
    weights_tensor = torch.FloatTensor(weights).to(device)
    
    print(f"Class weights calculated. Max: {weights.max():.2f}, Min: {weights.min():.2f}")
    return weights_tensor


def plot_curves(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    val_interval = history.get('val_interval', 1)
    val_epochs = [i * val_interval for i in range(1, len(history['val_r50']) + 1)]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    if len(history['val_r50']) > 0:
        plt.plot(val_epochs, history['val_r50'], 'g-o', label='Validation R@50')
    plt.title('Validation Recall@50')
    plt.xlabel('Epochs')
    plt.ylabel('R@50')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Curves updated: {save_path}")

def calculate_sgg_metrics(logits, rel_labels, k_list=[50, 100]):
    gt_indices = (rel_labels > 0).nonzero(as_tuple=True)[0]
    gt_count = len(gt_indices)
    
    gt_per_class = {}
    gt_triplets = set()
    
    for idx in gt_indices:
        pair_idx = idx.item()
        label = rel_labels[idx].item()
        gt_triplets.add((pair_idx, label))
        gt_per_class[label] = gt_per_class.get(label, 0) + 1

    if gt_count == 0:
        return {k: 0 for k in k_list}, 0, {k: {} for k in k_list}, {}

    scores_fg = logits[:, 1:] 
    num_pairs, num_fg_classes = scores_fg.shape
    flat_scores = scores_fg.flatten()
    max_k = max(k_list)
    current_k = min(flat_scores.shape[0], max_k)
    
    topk_vals, topk_inds = torch.topk(flat_scores, current_k)
    topk_pair_indices = (topk_inds // num_fg_classes).tolist()
    topk_labels = (topk_inds % num_fg_classes + 1).tolist()
    
    top_triplets = list(zip(topk_pair_indices, topk_labels))

    hits_k = {k: 0 for k in k_list}
    hits_per_class_k = {k: {} for k in k_list}

    for k in k_list:
        current_top = set(top_triplets[:k])
        matches = gt_triplets.intersection(current_top)
        hits_k[k] = len(matches)
        for (_, label) in matches:
            hits_per_class_k[k][label] = hits_per_class_k[k].get(label, 0) + 1
            
    return hits_k, gt_count, hits_per_class_k, gt_per_class

def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.overrides:
        cfg = update_config(cfg, args.overrides)

    set_seed(cfg["TRAIN"]["SEED"])

    if cfg["DATASET"]["USE_SYNTHETIC"]:
        train_ds = SyntheticSGGDataset(
            num_images=int(cfg["DATASET"]["NUM_SYNTHETIC_IMAGES"]),
            feat_dim=int(cfg["DATASET"]["FEAT_DIM"]),
            num_obj_classes=int(cfg["DATASET"]["NUM_OBJ_CLASSES"]),
            num_rel_classes=int(cfg["DATASET"]["NUM_REL_CLASSES"]),
            seed=cfg["TRAIN"]["SEED"],
        )
        val_ds = SyntheticSGGDataset(
            num_images=max(50, int(cfg["DATASET"]["NUM_SYNTHETIC_IMAGES"] // 5)),
            feat_dim=int(cfg["DATASET"]["FEAT_DIM"]),
            num_obj_classes=int(cfg["DATASET"]["NUM_OBJ_CLASSES"]),
            num_rel_classes=int(cfg["DATASET"]["NUM_REL_CLASSES"]),
            seed=cfg["TRAIN"]["SEED"] + 1,
        )
    else:
        data_dir = cfg["DATASET"]["DATA_DIR"]
        json_path = os.path.join(cfg["BASE_DIR"], 'image_data.json')
        h5_path = os.path.join(cfg["BASE_DIR"], 'VG-SGG-with-attri.h5')
        train_ds = NPZSceneGraphDataset(data_dir, image_data_json=json_path, h5_path=h5_path, split="train")
        val_ds = NPZSceneGraphDataset(data_dir, image_data_json=json_path, h5_path=h5_path, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["TRAIN"]["BATCH_SIZE"],
        shuffle=True, num_workers=cfg["TRAIN"]["NUM_WORKERS"],
        collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=cfg["TRAIN"]["NUM_WORKERS"],
        collate_fn=collate_fn, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CFEN(
        feat_dim=int(cfg["MODEL"]["FEAT_DIM"]),
        num_obj_classes=int(cfg["DATASET"]["NUM_OBJ_CLASSES"]),
        num_rel_classes=int(cfg["DATASET"]["NUM_REL_CLASSES"]),
        dm_lambda=float(cfg["MODEL"]["DM_LAMBDA"]),
        ema_alpha=float(cfg["MODEL"]["EMA_ALPHA"]),
        fusion=str(cfg["MODEL"]["FUSION"]),
    ).to(device)

    if not cfg["DATASET"]["USE_SYNTHETIC"]:
        h5_path = os.path.join(cfg["BASE_DIR"], 'VG-SGG-with-attri.h5')
        class_weights = get_class_weights(h5_path, device)
        weighted_criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        weighted_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["TRAIN"]["LR"],
        momentum=cfg["TRAIN"]["MOMENTUM"],
        weight_decay=cfg["TRAIN"]["WEIGHT_DECAY"]
    )

    start_epoch = 1
    history = {
        "train_loss": [],
        "val_r50": [],
        "val_interval": cfg["TRAIN"]["VAL_INTERVAL"]
    }

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'...")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
            if 'history' in checkpoint:
                history = checkpoint['history']
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    os.makedirs(cfg["OUTPUT"]["CKPT_DIR"], exist_ok=True)

    global_step = 0
    for epoch in range(start_epoch, cfg["TRAIN"]["EPOCHS"] + 1):
        model.train()
        
        epoch_loss_sum = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['TRAIN']['EPOCHS']}")
        for feats, obj_labels, rel_pairs, rel_labels, _ in pbar:
            if rel_pairs.shape[0] == 0:
                continue
                
            feats = feats.to(device)
            obj_labels = obj_labels.to(device)
            rel_pairs = rel_pairs.to(device)
            rel_labels = rel_labels.to(device)

            out = model(feats, obj_labels, rel_pairs, rel_labels, update_ema=True)
            
            loss_ce = weighted_criterion(out["logits"], rel_labels)
            loss_dm = out.get("loss_dm", torch.tensor(0.0, device=device))
            dm_lambda = float(cfg["MODEL"]["DM_LAMBDA"])
            
            loss = loss_ce + (dm_lambda * loss_dm)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["TRAIN"]["GRAD_CLIP_NORM"])
            optimizer.step()

            epoch_loss_sum += loss.item()
            num_batches += 1

            if global_step % cfg["TRAIN"]["LOG_INTERVAL"] == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "ce": f"{loss_ce.item():.4f}",
                    "dm": f"{loss_dm.item():.4f}"
                })
            global_step += 1

        avg_train_loss = epoch_loss_sum / max(1, num_batches)
        history['train_loss'].append(avg_train_loss)
        
        ckpt_path = os.path.join(cfg["OUTPUT"]["CKPT_DIR"], f"cfen_epoch{epoch}.pt")

        if epoch % cfg["TRAIN"]["VAL_INTERVAL"] == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": cfg,
                "history": history
            }, ckpt_path)

        if epoch % cfg["TRAIN"]["VAL_INTERVAL"] == 0:
            model.eval()
            
            total_hits = {50: 0, 100: 0}
            total_gt_count = 0
            class_hits = {50: {}, 100: {}}
            class_gt_count = {}
            
            with torch.no_grad():
                for i, (feats, obj_labels, rel_pairs, rel_labels, _) in enumerate(tqdm(val_loader, desc="Validating")):
                    if rel_pairs.shape[0] == 0:
                        continue
                    feats = feats.to(device)
                    obj_labels = obj_labels.to(device)
                    rel_pairs = rel_pairs.to(device)
                    rel_labels = rel_labels.to(device)
                    
                    out = model(feats, obj_labels, rel_pairs, None, update_ema=False)
                    logits = out["logits"]
                    
                    b_hits, b_gt, b_class_hits, b_class_gt = calculate_sgg_metrics(
                        logits, rel_labels, k_list=[50, 100]
                    )
                    
                    total_gt_count += b_gt
                    for k in [50, 100]:
                        total_hits[k] += b_hits[k]
                    
                    for cls, count in b_class_gt.items():
                        class_gt_count[cls] = class_gt_count.get(cls, 0) + count
                        
                    for k in [50, 100]:
                        for cls, hits in b_class_hits[k].items():
                            class_hits[k][cls] = class_hits[k].get(cls, 0) + hits

            r50 = total_hits[50] / max(1, total_gt_count)
            r100 = total_hits[100] / max(1, total_gt_count)
            
            mr_values = {50: [], 100: []}
            for cls in class_gt_count:
                gt_cnt = class_gt_count[cls]
                if gt_cnt > 0:
                    mr_values[50].append(class_hits[50].get(cls, 0) / gt_cnt)
                    mr_values[100].append(class_hits[100].get(cls, 0) / gt_cnt)
            
            mr50 = sum(mr_values[50]) / len(mr_values[50]) if mr_values[50] else 0.0
            mr100 = sum(mr_values[100]) / len(mr_values[100]) if mr_values[100] else 0.0

            print(f"\n[Val] Epoch {epoch} SGG Metrics:")
            print(f"  R@50:  {r50:.4f} | R@100:  {r100:.4f}")
            print(f"  mR@50: {mr50:.4f} | mR@100: {mr100:.4f}\n")
            
            history['val_r50'].append(r50)
            plot_curves(history, cfg["OUTPUT"]["CKPT_DIR"])

if __name__ == "__main__":
    main()