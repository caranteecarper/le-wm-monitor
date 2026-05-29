import os
from functools import partial
from pathlib import Path
import time

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf, open_dict

from kan_module import save_pred_proj_stats
from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def patch_vit_hf_imports():
    """Patch stable_pretraining's optional ViT imports when transformers is present."""

    try:
        from transformers import ViTConfig, ViTModel
        import stable_pretraining.backbone.utils as backbone_utils
    except Exception:
        return

    backbone_utils.ViTConfig = ViTConfig
    backbone_utils.ViTModel = ViTModel
    backbone_utils._TRANSFORMERS_AVAILABLE = True


def patch_stable_pretraining_single_optimizer():
    """兼容当前华为环境中 stable_pretraining 对单 optimizer 的处理。"""

    import stable_pretraining.module as spt_module
    from prettytable import PrettyTable

    def on_train_start(self):
        spt_module.logging.info("Double checking optimizers!")
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        spt_module.logging.info(f"`self.optimizers() gave us {len(optimizers)} optimizers")
        for i, optimizer in enumerate(optimizers):
            if i not in self._optimizer_index_to_name:
                name = f"default_{i}"
                self._optimizer_index_to_name[i] = name
            name = self._optimizer_index_to_name[i]
            if name not in self._optimizer_gradient_clip_val:
                spt_module.logging.warning(f"No clip val found for optimizer {name}")
                clip_val = getattr(
                    self.trainer, "gradient_clip_val_", self.trainer.gradient_clip_val
                )
                spt_module.logging.warning(f"-> we will use the Trainer's value of {clip_val}")
                self._optimizer_gradient_clip_val[name] = clip_val
            if name not in self._optimizer_gradient_clip_algorithm:
                spt_module.logging.warning(f"No clip algorithm found for optimizer {name}")
                clip_algo = getattr(
                    self.trainer,
                    "gradient_clip_algorithm_",
                    self.trainer.gradient_clip_algorithm,
                )
                spt_module.logging.warning(f"-> we will use the Trainer's value of {clip_algo}")
                self._optimizer_gradient_clip_algorithm[name] = clip_algo
            if name not in self._optimizer_frequencies:
                freq = getattr(self.trainer, "accumulate_grad_batches", 1)
                freq = getattr(self.trainer, "accumulate_grad_batches_", freq)
                freq = max(int(freq), 1)
                freq = self.optim.get("frequency", freq)
                self._optimizer_frequencies[name] = int(freq)

        table = PrettyTable()
        table.field_names = ["Opt. Index", "Opt. name", "opt", "clip val.", "clip alg."]
        for i, optimizer in enumerate(optimizers):
            name = self._optimizer_index_to_name[i]
            table.add_row(
                [
                    str(i),
                    name,
                    type(optimizer).__name__,
                    str(self._optimizer_gradient_clip_val[name]),
                    str(self._optimizer_gradient_clip_algorithm[name]),
                ]
            )
        spt_module.logging.success(
            "We are done checking your optimizers! Here is the summary:\n{}", table
        )

    spt_module.Module.on_train_start = on_train_start


def load_train_dataset(dataset_name, cache_dir=None, **dataset_cfg):
    """Load the official dataset, with an HDF5 fallback for older stable_worldmodel."""

    if hasattr(swm.data, "load_dataset"):
        return swm.data.load_dataset(
            dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
        )

    dataset_name = str(dataset_name)
    if dataset_name.endswith(".h5"):
        dataset_name = dataset_name[:-3]
    return swm.data.HDF5Dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )


def get_checkpoint_root() -> Path:
    try:
        return Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"))
    except TypeError:
        return Path(swm.data.utils.get_cache_dir()) / "checkpoints"


def patch_transform_compat(transform):
    """Patch old stable_pretraining transform instances on this environment."""

    def walk(obj, seen=None):
        if seen is None:
            seen = set()
        if id(obj) in seen:
            return
        seen.add(id(obj))
        yield obj
        for attr in ("args", "transforms"):
            children = getattr(obj, attr, None)
            if children is not None:
                for child in children:
                    yield from walk(child, seen)
        child = getattr(obj, "t", None)
        if child is not None:
            yield from walk(child, seen)
        for child in getattr(obj, "_modules", {}).values():
            if child is not None:
                yield from walk(child, seen)

    for module in walk(transform):
        if hasattr(module, "_transform") and not hasattr(module, "transform"):
            object.__setattr__(module, "transform", module._transform)
    return transform


def _last_dim_stats(prefix, x):
    x = x.detach().float()
    flat = x.reshape(-1, x.shape[-1]) if x.ndim > 1 else x.reshape(-1, 1)
    return {
        f"{prefix}_std": flat.std(unbiased=False),
        f"{prefix}_dim_std_min": flat.std(dim=0, unbiased=False).min(),
        f"{prefix}_dim_std_mean": flat.std(dim=0, unbiased=False).mean(),
    }


def build_monitor_dict(batch, emb, pred_emb, tgt_emb):
    with torch.no_grad():
        pred_flat = pred_emb.detach().float().reshape(-1, pred_emb.shape[-1])
        tgt_flat = tgt_emb.detach().float().reshape(-1, tgt_emb.shape[-1])
        metrics = {}
        metrics.update(_last_dim_stats("emb", emb))
        metrics.update(_last_dim_stats("pred_emb", pred_emb))
        metrics.update(_last_dim_stats("tgt_emb", tgt_emb))
        metrics["pred_tgt_cosine"] = F.cosine_similarity(
            pred_flat, tgt_flat, dim=-1
        ).mean()
        metrics["pred_tgt_l2"] = (pred_flat - tgt_flat).norm(dim=-1).mean()
        if "action" in batch:
            action = batch["action"].detach().float()
            metrics["action_nan_frac"] = torch.isnan(action).float().mean()
        return metrics


class ThroughputCallback(pl.Callback):
    """Log epoch time, throughput, and GPU memory without changing training."""

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - getattr(self, "_epoch_start", time.time())
        steps = max(int(trainer.num_training_batches), 1)
        metrics = {
            "train/epoch_time_sec": torch.tensor(elapsed, device=pl_module.device),
            "train/steps_per_sec": torch.tensor(steps / max(elapsed, 1e-6), device=pl_module.device),
        }
        if torch.cuda.is_available():
            metrics["train/max_gpu_mem_gb"] = torch.tensor(
                torch.cuda.max_memory_allocated() / (1024 ** 3),
                device=pl_module.device,
            )
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

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
    if hasattr(self.model.pred_proj, "sparse_reg_loss"):
        output["sparse_reg_loss"] = self.model.pred_proj.sparse_reg_loss()
        output["loss"] = output["loss"] + output["sparse_reg_loss"]
        output["projection_l1"] = self.model.pred_proj.projection_l1().detach()
        output["projection_sparsity"] = self.model.pred_proj.projection_sparsity().detach()
    output["total_loss"] = output["loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)

    sparse_metrics = {
        f"{stage}/{k}": v for k, v in output.items() if k in {"projection_l1", "projection_sparsity"}
    }
    if sparse_metrics:
        self.log_dict(sparse_metrics, on_step=True, on_epoch=True, sync_dist=True)

    monitor_cfg = cfg.get("monitor", {})
    if monitor_cfg.get("enabled", False):
        monitor_dict = build_monitor_dict(batch, emb, pred_emb, tgt_emb)
        monitor_dict = {f"{stage}/monitor/{k}": v.detach() for k, v in monitor_dict.items()}
        self.log_dict(
            monitor_dict,
            on_step=monitor_cfg.get("on_step", True),
            on_epoch=monitor_cfg.get("on_epoch", True),
            sync_dist=True,
        )
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    patch_vit_hf_imports()
    patch_stable_pretraining_single_optimizer()

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

    transform = patch_transform_compat(spt.data.transforms.Compose(*transforms))
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
    run_id = cfg.get("subdir") or ""
    run_dir = get_checkpoint_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    param_stats = save_pred_proj_stats(world_model, cfg, run_dir)
    print("Pred-proj parameter stats:")
    for key, value in param_stats.items():
        print(f"  {key}: {value}")

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

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    elif cfg.get("monitor", {}).get("enabled", False) and cfg.monitor.get("csv_logger", False):
        logger = CSVLogger(save_dir=str(run_dir), name="metrics")

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, ThroughputCallback()],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        seed=cfg.seed,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
