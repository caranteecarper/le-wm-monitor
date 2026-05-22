import argparse
import csv
import json
import os
from pathlib import Path
import random

import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

# Keep this symbol visible for object checkpoints saved while train.py was __main__.
from module import VelocityAuxHead
from utils import get_img_preprocessor


class MLPProbe(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe whether LeWM Reacher latents contain physical observation information."
    )
    cache_dir = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))
    parser.add_argument("--dataset", default="reacher")
    parser.add_argument(
        "--feature-mode",
        choices=["single_emb", "delta_concat"],
        default="single_emb",
        help=(
            "Probe single-frame embeddings or concat(emb_t, emb_t - emb_{t-1}) "
            "for temporal-delta velocity analysis."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(cache_dir / "reacher" / "lewm-monitor-3ep_object.ckpt"),
    )
    parser.add_argument(
        "--base-object-checkpoint",
        default=None,
        help=(
            "Optional serialized model object used when --checkpoint points to a "
            "weights-only state_dict such as *_weights_epoch_1.pt."
        ),
    )
    parser.add_argument("--cache-dir", default=str(cache_dir))
    parser.add_argument(
        "--out-dir",
        default=str(cache_dir / "probes" / "reacher_lewm-monitor-3ep"),
    )
    parser.add_argument("--max-sequences", type=int, default=10000)
    parser.add_argument("--extract-batch-size", type=int, default=64)
    parser.add_argument("--probe-batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(args):
    checkpoint = Path(args.checkpoint)
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if isinstance(obj, nn.Module):
        return obj

    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object type: {type(obj)!r}")

    if args.base_object_checkpoint is None:
        raise ValueError(
            "--checkpoint appears to be a weights-only state_dict. "
            "Pass --base-object-checkpoint pointing to a matching *_object.ckpt file."
        )

    print("loading base model object:", args.base_object_checkpoint)
    model = torch.load(args.base_object_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(obj, strict=True)
    return model


def load_dataset(args):
    dataset = swm.data.HDF5Dataset(
        args.dataset,
        num_steps=4,
        frameskip=5,
        keys_to_load=["pixels", "observation"],
        keys_to_cache=["observation"],
        cache_dir=args.cache_dir,
    )
    dataset.transform = spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=224)
    )
    return dataset


def build_probe_features(args, emb, obs):
    if args.feature_mode == "single_emb":
        return emb.reshape(-1, emb.shape[-1]), obs.reshape(-1, obs.shape[-1])

    if emb.size(1) < 2:
        raise ValueError("feature-mode=delta_concat requires at least two time steps.")

    emb_t = emb[:, 1:]
    emb_delta = emb[:, 1:] - emb[:, :-1]
    features = torch.cat((emb_t, emb_delta), dim=-1)
    targets = obs[:, 1:]
    return features.reshape(-1, features.shape[-1]), targets.reshape(-1, targets.shape[-1])


def extract_latents(args, model, dataset):
    rng = random.Random(args.seed)
    n = min(args.max_sequences, len(dataset))
    indices = sorted(rng.sample(range(len(dataset)), n))
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.extract_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    xs, ys = [], []
    model.eval()
    model.requires_grad_(False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            pixels = batch["pixels"].to(args.device, non_blocking=True)
            output = model.encode({"pixels": pixels})
            emb = output["emb"].detach().cpu()
            obs = batch["observation"].detach().float().cpu()
            x, y = build_probe_features(args, emb, obs)
            xs.append(x)
            ys.append(y)
            if (batch_idx + 1) % 25 == 0:
                print(f"extracted {min((batch_idx + 1) * args.extract_batch_size, n)}/{n} sequences")

    x = torch.cat(xs, dim=0).float()
    y = torch.cat(ys, dim=0).float()
    valid = torch.isfinite(x).all(dim=1) & torch.isfinite(y).all(dim=1)
    x = x[valid]
    y = y[valid]
    return x, y


def standardize(train, val):
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train - mean) / std, (val - mean) / std, mean, std


def split_data(x, y, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)
    n_train = int(0.8 * x.shape[0])
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def pearson_corr(pred, target):
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)
    num = (pred_c * target_c).mean(dim=0)
    den = pred_c.std(dim=0).clamp_min(1e-8) * target_c.std(dim=0).clamp_min(1e-8)
    corr = num / den
    return corr


def train_probe(args, name, probe, x_train, y_train, x_val, y_val, y_mean, y_std, out_dir):
    probe = probe.to(args.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.probe_batch_size,
        shuffle=True,
        num_workers=0,
    )

    x_val_d = x_val.to(args.device)
    y_val_d = y_val.to(args.device)
    best = {"val_mse_norm": float("inf"), "epoch": -1}
    history = []

    for epoch in range(args.epochs):
        probe.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            pred = probe(xb)
            loss = (pred - yb).pow(2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu())

        probe.eval()
        with torch.no_grad():
            val_pred = probe(x_val_d)
            val_mse_norm = (val_pred - y_val_d).pow(2).mean().item()
            train_mse_norm = torch.stack(losses).mean().item()

        row = {
            "probe": name,
            "epoch": epoch + 1,
            "train_mse_norm": train_mse_norm,
            "val_mse_norm": val_mse_norm,
        }
        history.append(row)
        if val_mse_norm < best["val_mse_norm"]:
            best = row
            torch.save(probe.state_dict(), out_dir / f"{name}_probe.pt")

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"{name} epoch {epoch + 1}/{args.epochs}: val_mse_norm={val_mse_norm:.6f}")

    probe.load_state_dict(torch.load(out_dir / f"{name}_probe.pt", map_location=args.device))
    probe.eval()
    with torch.no_grad():
        pred_norm = probe(x_val_d).cpu()
    pred = pred_norm * y_std + y_mean
    target = y_val * y_std + y_mean
    mse_per_dim = (pred - target).pow(2).mean(dim=0)
    corr_per_dim = pearson_corr(pred, target)

    result = {
        "probe": name,
        "best_epoch": int(best["epoch"]),
        "val_mse_norm": float(best["val_mse_norm"]),
        "val_mse_original_mean": float(mse_per_dim.mean()),
        "pearson_mean": float(corr_per_dim.mean()),
        "mse_per_dim": [float(v) for v in mse_per_dim],
        "pearson_per_dim": [float(v) for v in corr_per_dim],
    }
    return result, history


def write_outputs(out_dir, args, results, histories, x, y):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "probe_results.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, ensure_ascii=False, indent=2)

    with (out_dir / "probe_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["probe", "best_epoch", "val_mse_norm", "val_mse_original_mean", "pearson_mean"])
        for r in results:
            writer.writerow(
                [
                    r["probe"],
                    r["best_epoch"],
                    r["val_mse_norm"],
                    r["val_mse_original_mean"],
                    r["pearson_mean"],
                ]
            )

    with (out_dir / "probe_history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["probe", "epoch", "train_mse_norm", "val_mse_norm"]
        )
        writer.writeheader()
        for row in histories:
            writer.writerow(row)

    lines = [
        "# Reacher Latent Probing 结果",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- dataset: `{args.dataset}`",
        f"- feature mode: `{args.feature_mode}`",
        f"- 提取序列数: `{args.max_sequences}`",
        f"- 有效 latent 帧数: `{x.shape[0]}`",
        f"- latent 维度: `{x.shape[1]}`",
        f"- observation 维度: `{y.shape[1]}`",
        "",
        "## 指标含义",
        "",
        "- `val_mse_norm`: 标准化 observation 上的验证 MSE，越低越好。",
        "- `pearson_mean`: 每个 observation 维度 Pearson 相关系数的平均值，越接近 1 越好。",
        "- 如果 probe 表现好，说明 LeWM latent 中包含可恢复的物理状态信息。",
        "",
        "## 汇总",
        "",
        "| probe | best_epoch | val_mse_norm | val_mse_original_mean | pearson_mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['probe']} | {r['best_epoch']} | {r['val_mse_norm']:.6f} | "
            f"{r['val_mse_original_mean']:.6f} | {r['pearson_mean']:.6f} |"
        )
    (out_dir / "probe_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading model:", args.checkpoint)
    model = load_model(args)
    model = model.to(args.device)

    print("loading dataset:", args.dataset)
    dataset = load_dataset(args)
    print("dataset length:", len(dataset))

    print("extracting latents")
    x, y = extract_latents(args, model, dataset)
    print("features:", tuple(x.shape), "targets:", tuple(y.shape))

    x_train, y_train, x_val, y_val = split_data(x, y, args.seed)
    x_train, x_val, x_mean, x_std = standardize(x_train, x_val)
    y_train, y_val, y_mean, y_std = standardize(y_train, y_val)

    results = []
    histories = []
    linear = nn.Linear(x_train.shape[1], y_train.shape[1])
    result, history = train_probe(
        args, "linear", linear, x_train, y_train, x_val, y_val, y_mean, y_std, out_dir
    )
    results.append(result)
    histories.extend(history)

    mlp = MLPProbe(x_train.shape[1], y_train.shape[1])
    result, history = train_probe(
        args, "mlp", mlp, x_train, y_train, x_val, y_val, y_mean, y_std, out_dir
    )
    results.append(result)
    histories.extend(history)

    write_outputs(out_dir, args, results, histories, x, y)
    print("saved outputs to:", out_dir)
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
