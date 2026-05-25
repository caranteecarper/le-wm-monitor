import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Make LeWM latent prediction videos using nearest-neighbor frames. "
            "The model has no pixel decoder, so predicted frames are retrieved "
            "from dataset frames whose latents are closest to predicted latents."
        )
    )
    parser.add_argument("--dataset", required=True, choices=["cartpole", "pendulum", "reacher"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", default="/root/autodl-tmp/stable-wm")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/stable-wm/dmc_prediction_videos")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--db-frames", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def preprocess_pixels(frames):
    x = torch.from_numpy(frames).float() / 255.0
    x = x.permute(0, 3, 1, 2)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def get_action_stats(h5):
    action = h5["action"][:]
    valid = np.isfinite(action).all(axis=1)
    action = action[valid]
    mean = action.mean(axis=0, keepdims=True)
    std = action.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_action(action, mean, std):
    action = np.nan_to_num(action, nan=0.0).astype(np.float32)
    return (action - mean) / std


@torch.no_grad()
def encode_frames(model, frames, device, batch_size):
    embs = []
    for i in range(0, len(frames), batch_size):
        pixels = preprocess_pixels(frames[i : i + batch_size]).to(device)
        out = model.encode({"pixels": pixels.unsqueeze(1)})
        embs.append(out["emb"][:, 0].detach().cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def predict_latents(model, true_emb, action, history_size, horizon, device):
    emb = true_emb[:history_size].unsqueeze(0).to(device)
    act = action[:history_size].unsqueeze(0).to(device)
    preds = []

    for t in range(horizon):
        act_emb = model.action_encoder(act[:, -history_size:])
        pred = model.predict(emb[:, -history_size:], act_emb)[:, -1]
        preds.append(pred.detach().cpu())
        emb = torch.cat([emb, pred.unsqueeze(1)], dim=1)

        next_action_idx = history_size + t
        if next_action_idx < action.shape[0]:
            next_action = action[next_action_idx : next_action_idx + 1].to(device)
        else:
            next_action = torch.zeros_like(action[:1]).to(device)
        act = torch.cat([act, next_action.unsqueeze(0)], dim=1)

    return torch.cat(preds, dim=0)


def add_label(frame, label):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 224, 22), fill=(0, 0, 0))
    draw.text((6, 4), label, fill=(255, 255, 255))
    return np.asarray(img)


def side_by_side(real, pred, t):
    real = add_label(real, f"real future t+{t}")
    pred = add_label(pred, f"nearest predicted latent t+{t}")
    gap = np.full((real.shape[0], 8, 3), 255, dtype=np.uint8)
    return np.concatenate([real, gap, pred], axis=1)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_path = cache_dir / f"{args.dataset}.h5"
    checkpoint = Path(args.checkpoint)
    print("loading model:", checkpoint)
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = model.to(args.device).eval()
    model.requires_grad_(False)

    print("loading dataset:", h5_path)
    with h5py.File(h5_path, "r") as h5:
        action_mean, action_std = get_action_stats(h5)

        total = h5["pixels"].shape[0]
        db_count = min(args.db_frames, total)
        db_indices = np.sort(rng.choice(total, size=db_count, replace=False))
        print("encoding nearest-neighbor database:", db_count, "frames")
        db_frames = h5["pixels"][db_indices]
        db_emb = encode_frames(model, db_frames, args.device, args.batch_size)
        db_emb = F.normalize(db_emb.float(), dim=-1)

        ep_offset = h5["ep_offset"][:]
        ep_len = h5["ep_len"][:]

        for ep in args.episodes:
            if ep >= len(ep_len):
                print("skip missing episode:", ep)
                continue

            start = int(ep_offset[ep])
            length = int(ep_len[ep])
            needed = args.history_size + args.horizon
            if length < needed:
                print("skip short episode:", ep, "length:", length)
                continue

            raw_frames = h5["pixels"][start : start + needed]
            raw_action = h5["action"][start : start + needed]
            action = normalize_action(raw_action, action_mean, action_std)
            action = torch.from_numpy(action).float()

            true_emb = encode_frames(model, raw_frames, args.device, args.batch_size)
            pred_emb = predict_latents(
                model,
                true_emb,
                action,
                args.history_size,
                args.horizon,
                args.device,
            )
            pred_emb = F.normalize(pred_emb.float(), dim=-1)

            sims = pred_emb @ db_emb.T
            nn_pos = sims.argmax(dim=1).cpu().numpy()
            pred_frames = db_frames[nn_pos]
            real_frames = raw_frames[args.history_size : args.history_size + args.horizon]

            video = [
                side_by_side(real_frames[i], pred_frames[i], i + 1)
                for i in range(args.horizon)
            ]
            out = out_dir / f"{args.dataset}_latent_nn_ep{ep}.mp4"
            imageio.mimsave(out, video, fps=args.fps)
            print("saved:", out)


if __name__ == "__main__":
    main()
