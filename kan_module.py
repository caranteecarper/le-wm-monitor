"""KAN modules for the pred_proj-only ablation.

本文件只提供 pred_proj 替换模块，不改 LeWM 的 encoder、predictor、
attention、AdaLN 或 SIGReg。默认配置仍然会走官方 MLP。
"""

import csv
import json
from pathlib import Path

import torch
from torch import nn


class GaussianRBF(nn.Module):
    """Fixed Gaussian RBF basis used by FastKANLinear."""

    def __init__(
        self,
        grid_size: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        adaptive_grid: bool = False,
    ):
        super().__init__()
        if adaptive_grid:
            raise NotImplementedError("Stage 1 keeps adaptive_grid disabled.")
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")
        grid = torch.linspace(grid_min, grid_max, grid_size)
        self.register_buffer("grid", grid)
        self.scale = 1.0 / max(float(grid[1] - grid[0]), 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = torch.exp(-((x.unsqueeze(-1) - self.grid) * self.scale).pow(2))
        return basis.flatten(start_dim=-2)


class FastKANLinear(nn.Module):
    """FastKAN-style linear layer with Gaussian RBF basis.

    普通线性层全部使用 torch.nn.Linear；不手写随机 Parameter 初始化。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        adaptive_grid: bool = False,
        use_base_linear: bool = True,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.rbf = GaussianRBF(
            grid_size=grid_size,
            grid_min=grid_min,
            grid_max=grid_max,
            adaptive_grid=adaptive_grid,
        )
        self.spline_linear = nn.Linear(in_features * grid_size, out_features, bias=False)
        self.base_linear = (
            nn.Linear(in_features, out_features) if use_base_linear else None
        )
        self.base_activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.spline_linear(self.rbf(x))
        if self.base_linear is not None:
            out = out + self.base_linear(self.base_activation(x))
        return out


class SparseBottleneckFastKAN(nn.Module):
    """Sparse bottleneck FastKAN pred_proj.

    结构:
      Linear(input_dim -> bottleneck_dim, bias=False)
      BatchNorm1d
      FastKANLinear(bottleneck_dim -> bottleneck_dim)
      Linear(bottleneck_dim -> output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        output_dim: int,
        grid_size: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        adaptive_grid: bool = False,
        use_base_linear: bool = True,
        sparse_lambda: float = 1e-5,
    ):
        super().__init__()
        self.sparse_lambda = float(sparse_lambda)
        self.projection = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.norm = nn.BatchNorm1d(bottleneck_dim)
        self.kan = FastKANLinear(
            bottleneck_dim,
            bottleneck_dim,
            grid_size=grid_size,
            grid_min=grid_min,
            grid_max=grid_max,
            adaptive_grid=adaptive_grid,
            use_base_linear=use_base_linear,
        )
        self.output = nn.Linear(bottleneck_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.norm(x)
        x = self.kan(x)
        return self.output(x)

    def sparse_reg_loss(self) -> torch.Tensor:
        return self.sparse_lambda * self.projection.weight.abs().mean()

    def projection_l1(self) -> torch.Tensor:
        return self.projection.weight.abs().mean()

    def projection_sparsity(self, threshold: float = 1e-3) -> torch.Tensor:
        weight = self.projection.weight.detach()
        return (weight.abs() < threshold).float().mean()

    def topk_input_dims(self, top_k: int = 10):
        weight = self.projection.weight.detach().abs().cpu()
        rows = []
        for factor_idx in range(weight.shape[0]):
            values, indices = torch.topk(weight[factor_idx], k=min(top_k, weight.shape[1]))
            rows.append(
                {
                    "factor": int(factor_idx),
                    "latent_dims": [int(i) for i in indices.tolist()],
                    "weights": [float(v) for v in values.tolist()],
                }
            )
        return rows


def build_pred_proj(
    pred_proj_type: str = "mlp",
    input_dim: int = 192,
    output_dim: int = 192,
    hidden_dim: int = 2048,
    bottleneck_dim: int = 64,
    grid_size: int = 8,
    grid_min: float = -2.0,
    grid_max: float = 2.0,
    adaptive_grid: bool = False,
    use_base_linear: bool = True,
    sparse_lambda: float = 1e-5,
):
    """Build pred_proj for stage-1 component-wise replacement."""

    if pred_proj_type == "mlp":
        from module import MLP

        return MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            norm_fn=nn.BatchNorm1d,
        )

    if pred_proj_type == "direct_fastkan":
        return nn.Sequential(
            nn.BatchNorm1d(input_dim),
            FastKANLinear(
                input_dim,
                output_dim,
                grid_size=grid_size,
                grid_min=grid_min,
                grid_max=grid_max,
                adaptive_grid=adaptive_grid,
                use_base_linear=use_base_linear,
            ),
        )

    if pred_proj_type == "bottleneck_mlp_control":
        return nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, output_dim),
        )

    if pred_proj_type == "bottleneck_fastkan":
        return nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            FastKANLinear(
                bottleneck_dim,
                bottleneck_dim,
                grid_size=grid_size,
                grid_min=grid_min,
                grid_max=grid_max,
                adaptive_grid=adaptive_grid,
                use_base_linear=use_base_linear,
            ),
            nn.Linear(bottleneck_dim, output_dim),
        )

    if pred_proj_type == "sparse_bottleneck_fastkan":
        return SparseBottleneckFastKAN(
            input_dim=input_dim,
            bottleneck_dim=bottleneck_dim,
            output_dim=output_dim,
            grid_size=grid_size,
            grid_min=grid_min,
            grid_max=grid_max,
            adaptive_grid=adaptive_grid,
            use_base_linear=use_base_linear,
            sparse_lambda=sparse_lambda,
        )

    raise ValueError(f"Unsupported pred_proj_type={pred_proj_type!r}")


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_fastkan_params(module: nn.Module) -> int:
    return sum(
        count_trainable_params(submodule)
        for submodule in module.modules()
        if isinstance(submodule, FastKANLinear)
    )


def count_bottleneck_projection_params(module: nn.Module, bottleneck_dim: int) -> int:
    if isinstance(module, SparseBottleneckFastKAN):
        return count_trainable_params(module.projection)
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear) and submodule.out_features == bottleneck_dim:
            return count_trainable_params(submodule)
    return 0


def pred_proj_stats(model: nn.Module, cfg) -> dict:
    pred_proj = model.pred_proj
    bottleneck_dim = int(cfg.model.get("bottleneck_dim", 64))
    return {
        "variant": cfg.model.get("pred_proj_type", "mlp"),
        "total_trainable_params": count_trainable_params(model),
        "pred_proj_trainable_params": count_trainable_params(pred_proj),
        "fastkan_params": count_fastkan_params(pred_proj),
        "bottleneck_projection_params": count_bottleneck_projection_params(
            pred_proj, bottleneck_dim
        ),
        "sparse_extra_params": 0,
        "bottleneck_dim": bottleneck_dim,
        "grid_size": int(cfg.model.get("grid_size", 8)),
        "sparse_lambda": float(cfg.model.get("sparse_lambda", 1e-5)),
    }


def save_pred_proj_stats(model: nn.Module, cfg, run_dir: Path) -> dict:
    results_dir = Path(run_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stats = pred_proj_stats(model, cfg)
    with (results_dir / "param_count.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    with (results_dir / "param_count.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "total_trainable_params",
                "pred_proj_trainable_params",
                "fastkan_params",
                "bottleneck_projection_params",
                "sparse_extra_params",
                "bottleneck_dim",
                "grid_size",
                "sparse_lambda",
            ],
        )
        writer.writeheader()
        writer.writerow(stats)
    return stats
