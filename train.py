import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg, VelocityAuxHead
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def load_train_dataset(dataset_name, cache_dir=None, **dataset_cfg):
    if hasattr(swm.data, "load_dataset"):
        return swm.data.load_dataset(
            dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
        )

    dataset_name = str(dataset_name)
    if dataset_name.endswith(".h5"):
        dataset_name = dataset_name[:-3]
    if dataset_name.endswith(".lance"):
        raise ValueError(
            "stable-worldmodel==0.0.6 does not support Lance datasets. "
            "Use the HDF5 dataset instead, for example "
            "'pusht_expert_train.h5' under $STABLEWM_HOME."
        )

    return swm.data.HDF5Dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )


def _norm_mean(x):
    x = x.detach().float()
    return x.norm(dim=-1).mean()


def _last_dim_stats(prefix, x):
    x = x.detach().float()
    flat = x.reshape(-1, x.shape[-1]) if x.ndim > 1 else x.reshape(-1, 1)
    return {
        f"{prefix}_mean": flat.mean(),
        f"{prefix}_std": flat.std(unbiased=False),
        f"{prefix}_norm": flat.norm(dim=-1).mean(),
        f"{prefix}_dim_std_mean": flat.std(dim=0, unbiased=False).mean(),
        f"{prefix}_dim_std_min": flat.std(dim=0, unbiased=False).min(),
    }


def build_monitor_dict(batch, emb, act_emb, pred_emb, tgt_emb):
    with torch.no_grad():
        metrics = {}
        metrics.update(_last_dim_stats("emb", emb))
        metrics.update(_last_dim_stats("act_emb", act_emb))
        metrics.update(_last_dim_stats("pred_emb", pred_emb))
        metrics.update(_last_dim_stats("tgt_emb", tgt_emb))

        pred_flat = pred_emb.detach().float().reshape(-1, pred_emb.shape[-1])
        tgt_flat = tgt_emb.detach().float().reshape(-1, tgt_emb.shape[-1])
        metrics["pred_tgt_cosine"] = F.cosine_similarity(
            pred_flat, tgt_flat, dim=-1
        ).mean()
        metrics["pred_tgt_l2"] = (pred_flat - tgt_flat).norm(dim=-1).mean()

        if emb.size(1) > 1:
            metrics["emb_delta_norm"] = _norm_mean(emb[:, 1:] - emb[:, :-1])
        if emb.size(1) > 2:
            velocity = emb[:, 1:] - emb[:, :-1]
            v_prev = velocity[:, :-1].detach().float().reshape(-1, emb.shape[-1])
            v_next = velocity[:, 1:].detach().float().reshape(-1, emb.shape[-1])
            metrics["temporal_straightness"] = F.cosine_similarity(
                v_prev, v_next, dim=-1
            ).mean()
        if "action" in batch:
            action = batch["action"].detach().float()
            metrics["action_norm"] = _norm_mean(action)
            metrics["action_abs_mean"] = action.abs().mean()
        for key in ("state", "proprio", "observation"):
            if key in batch and torch.is_tensor(batch[key]):
                metrics[f"{key}_norm"] = _norm_mean(batch[key])

        return metrics


def get_velocity_aux_mode(velocity_cfg):
    mode = velocity_cfg.get("mode", "delta_concat")
    if mode not in {"single_emb", "delta_concat"}:
        raise ValueError(
            "Unsupported velocity_aux.mode="
            f"{mode!r}; expected 'single_emb' or 'delta_concat'."
        )
    return mode


def get_velocity_aux_input_dim(velocity_cfg, embed_dim):
    mode = get_velocity_aux_mode(velocity_cfg)
    return 2 * embed_dim if mode == "delta_concat" else embed_dim


def build_velocity_aux_batch(emb, observation, columns, mode):
    if mode == "single_emb":
        return emb, observation[..., columns]

    if emb.size(1) < 2:
        raise ValueError("velocity_aux.mode='delta_concat' requires at least two time steps.")

    emb_t = emb[:, 1:]
    emb_delta = emb[:, 1:] - emb[:, :-1]
    aux_input = torch.cat((emb_t, emb_delta), dim=-1)
    target = observation[:, 1:, columns]
    return aux_input, target


def add_velocity_aux_loss(self, output, batch, emb, stage, cfg):
    velocity_cfg = cfg.get("velocity_aux", {})
    if not velocity_cfg.get("enabled", False):
        return
    if "observation" not in batch:
        raise KeyError("velocity_aux requires batch['observation']; enable it in data keys_to_load.")
    if not hasattr(self.model, "velocity_aux_head"):
        raise AttributeError("velocity_aux is enabled but model.velocity_aux_head is missing.")

    columns = [int(c) for c in velocity_cfg.get("columns", [4, 5])]
    mode = get_velocity_aux_mode(velocity_cfg)
    aux_input, target = build_velocity_aux_batch(
        emb, batch["observation"], columns, mode
    )
    target = target.detach().float()
    pred = self.model.velocity_aux_head(aux_input).float()

    per_dim_mse = (pred - target).pow(2).mean(dim=(0, 1))
    aux_loss = per_dim_mse.mean()
    weight = float(velocity_cfg.get("weight", 0.05))
    weighted_loss = weight * aux_loss

    output["velocity_aux_loss"] = aux_loss
    output["velocity_aux_loss_weighted"] = weighted_loss
    output["loss"] = output["loss"] + weighted_loss

    metrics = {
        "loss": aux_loss.detach(),
        "loss_weighted": weighted_loss.detach(),
        "pred_std": pred.detach().std(unbiased=False),
        "target_std": target.detach().std(unbiased=False),
    }
    if mode == "delta_concat":
        metrics["emb_delta_norm"] = aux_input[..., emb.shape[-1] :].detach().norm(dim=-1).mean()
    for i, col in enumerate(columns):
        name = "qvel0" if col == 4 else "qvel1" if col == 5 else f"obs{col}"
        metrics[f"{name}_mse"] = per_dim_mse[i].detach()

    self.log_dict(
        {f"{stage}/velocity_aux/{k}": v for k, v in metrics.items()},
        on_step=velocity_cfg.get("on_step", True),
        on_epoch=velocity_cfg.get("on_epoch", True),
        sync_dist=True,
    )


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    action_nan_frac = torch.isnan(batch["action"]).float().mean()

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  
    add_velocity_aux_loss(self, output, batch, emb, stage, cfg)

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)

    monitor_cfg = cfg.get("monitor", {})
    if monitor_cfg.get("enabled", False):
        monitor_dict = build_monitor_dict(batch, emb, act_emb, pred_emb, tgt_emb)
        monitor_dict["action_nan_frac"] = action_nan_frac.detach()
        monitor_dict = {
            f"{stage}/monitor/{k}": v.detach() for k, v in monitor_dict.items()
        }
        self.log_dict(
            monitor_dict,
            on_step=monitor_cfg.get("on_step", True),
            on_epoch=monitor_cfg.get("on_epoch", True),
            sync_dist=True,
        )
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = load_train_dataset(dataset_name, cache_dir=cache_dir, **dataset_cfg)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)
    velocity_cfg = cfg.get("velocity_aux", {})
    if velocity_cfg.get("enabled", False):
        columns = [int(c) for c in velocity_cfg.get("columns", [4, 5])]
        world_model.velocity_aux_head = VelocityAuxHead(
            input_dim=get_velocity_aux_input_dim(velocity_cfg, cfg.wm.embed_dim),
            output_dim=len(columns),
            hidden_dim=int(velocity_cfg.get("hidden_dim", 256)),
        )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), "checkpoints", run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    elif cfg.monitor.enabled and cfg.monitor.get("csv_logger", False):
        logger = CSVLogger(save_dir=str(run_dir), name="metrics")

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
