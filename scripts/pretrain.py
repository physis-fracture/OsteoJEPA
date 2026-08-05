"""Stage A pretraining.

    L = L_pred + lambda_v * L_var + lambda_c * L_cov + lambda_m * L_margin

Only clean images from the training fold enter here. The model must never see
pathology during Stage A; that condition is what makes the residual meaningful.

Two monitors run every epoch and either can end the run early:

* Var(z) as the **minimum** across dimensions. A collapse confined to a few
  dimensions is invisible in the mean.
* Median V(p) = Var_a s(p, a). If it has not moved from its initialization value
  after `monitor.abort_if_V_flat_after_epoch` epochs, the predictor is ignoring
  the age channel, every Stage B quantity is meaningless, and the run stops
  rather than burning a day to find out later.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physis.data.dataset import PhysisDataset, load_manifest, select_split, subset_by_study
from physis.data.geometry import age_band_index, band_list
from physis.data.masking import build_batch_masks
from physis.eval.figures import plot_band_curves, plot_spatial_map, plot_training_curves
from physis.losses.margin import age_grid
from physis.losses.objective import compute_objective
from physis.models.osteojepa import OsteoJEPA, momentum_at
from physis.utils.config import load_config
from physis.utils.run import resolve_device, setup_run


class VFlatAbort(RuntimeError):
    """Raised when the pre-registered age-sensitivity criterion fires."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OsteoJEPA Stage A pretraining")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="key.sub=value overrides")
    parser.add_argument("--run-subdir", default=None)
    parser.add_argument("--resume", default=None, help="checkpoint to continue from")
    return parser.parse_args()


def lr_at(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def save_checkpoint(path: Path, model, optimizer, cfg, *, epoch: int, step: int, best: float) -> None:
    """Everything needed to continue: a rented GPU box can vanish mid-run."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": step,
            "best_loss": best,
            "config": {
                "encoder_depth": int(cfg.encoder.depth),
                "predictor_depth": int(cfg.predictor.depth),
            },
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    run = setup_run(cfg, subdir=args.run_subdir, reuse=bool(args.resume))
    log = run.log

    device = resolve_device(str(cfg.optim.device))
    use_amp = device.type == "cuda" and str(cfg.run.amp) == "bf16"
    log.info("device: %s (amp=%s)", device, use_amp)

    manifest = load_manifest(cfg)
    train_df = select_split(manifest, "train", clean_only=True)
    if int(cfg.data.subset_train) > 0:
        train_df = subset_by_study(train_df, int(cfg.data.subset_train), int(cfg.run.seed))
    log.info("train images: %d (clean, fold train)", len(train_df))
    assert len(train_df) > 0, "no training images selected"

    bands = band_list(cfg)
    band_of_row = {
        str(row.stem): age_band_index(float(row.age), bands) for row in train_df.itertuples()
    }

    dataset = PhysisDataset(train_df, cfg, augment=bool(cfg.data.augment), seed=int(cfg.run.seed))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.optim.batch_size),
        shuffle=True,
        num_workers=int(cfg.optim.num_workers),
        drop_last=len(dataset) >= int(cfg.optim.batch_size),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg.optim.num_workers) > 0,
    )

    model = OsteoJEPA(cfg).to(device)
    trainable = (
        list(model.encoder.parameters())
        + list(model.predictor.parameters())
        + list(model.condition.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable, lr=float(cfg.optim.lr), weight_decay=float(cfg.optim.weight_decay)
    )

    steps_per_epoch = max(len(loader), 1)
    epochs = int(cfg.optim.epochs)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * int(cfg.optim.warmup_epochs)
    log.info("steps: %d per epoch, %d total", steps_per_epoch, total_steps)

    start_epoch, global_step, best_loss = 0, 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_loss = float(state.get("best_loss", float("inf")))
        log.info("resumed from %s at epoch %d, step %d", args.resume, start_epoch, global_step)

    grid = age_grid(
        float(cfg.inference.age_min),
        float(cfg.inference.age_max),
        float(cfg.inference.age_grid_fine_step),
    )
    rng = np.random.default_rng(int(cfg.run.seed) + start_epoch)
    lambda_margin = float(cfg.loss.lambda_margin)
    n_tokens = model.num_patches
    checkpoint_every = int(cfg.run.checkpoint_every)
    v_first_epoch: float | None = None
    lr = float(cfg.optim.lr)

    for epoch in range(start_epoch, epochs):
        model.train()
        totals = {k: 0.0 for k in ("loss", "pred", "var", "cov", "margin")}
        v_medians, std_mins, step_times = [], [], []
        v_by_band: dict[int, list[float]] = {}
        v_spatial_sum = np.zeros(n_tokens, dtype=np.float64)
        v_spatial_count = np.zeros(n_tokens, dtype=np.int64)

        for batch in loader:
            step_start = time.perf_counter()
            lr = lr_at(
                global_step, total_steps, warmup_steps,
                float(cfg.optim.lr), float(cfg.optim.min_lr),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            images = batch["image"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True)
            age = batch["age"].to(device)
            meta = {k: batch[k].to(device) for k in ("gender", "view", "laterality")}
            masks = build_batch_masks(valid, cfg, rng)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = compute_objective(
                    model,
                    images=images,
                    ages=age,
                    ages_cpu=batch["age"].numpy(),
                    meta=meta,
                    masks=masks,
                    cfg=cfg,
                    rng=rng,
                    grid=grid,
                    use_margin=lambda_margin > 0,
                )

            loss = out.total
            assert torch.isfinite(loss), f"non-finite loss at step {global_step}"
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(cfg.optim.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(trainable, float(cfg.optim.grad_clip))
            optimizer.step()
            model.ema_update(
                momentum_at(
                    global_step, total_steps,
                    float(cfg.target_encoder.ema_start), float(cfg.target_encoder.ema_end),
                )
            )

            totals["loss"] += float(loss.detach())
            totals["pred"] += float(out.pred.detach())
            totals["var"] += float(out.var.detach())
            totals["cov"] += float(out.cov.detach())
            totals["margin"] += float(out.margin.detach())
            if out.std_per_dim.numel():
                std_mins.append(float(out.std_per_dim.min()))

            if out.v_patch is not None:
                _accumulate_age_sensitivity(
                    out, masks, batch, band_of_row, v_medians, v_by_band,
                    v_spatial_sum, v_spatial_count,
                )

            step_times.append(time.perf_counter() - step_start)
            global_step += 1
            if global_step % int(cfg.run.log_every) == 0:
                log.info(
                    "step %d/%d loss %.4f lr %.2e",
                    global_step, total_steps, float(loss.detach()), lr,
                )

        epoch_loss = totals["loss"] / steps_per_epoch
        record = {
            "epoch": epoch,
            "lr": lr,
            **{f"loss_{k}": v / steps_per_epoch for k, v in totals.items()},
            "var_z_min_dim": float(np.min(std_mins) ** 2) if std_mins else float("nan"),
            "v_p_median": float(np.median(v_medians)) if v_medians else float("nan"),
            "v_p_by_band": {
                bands[index]["name"]: float(np.median(values))
                for index, values in sorted(v_by_band.items())
            },
            "sec_per_step": float(np.mean(step_times)),
            "images_per_sec": float(int(cfg.optim.batch_size) / np.mean(step_times)),
        }
        run.append_metrics(record)
        # V(p) and Var(z) start many orders of magnitude below 1, and the whole
        # point of logging them is to watch them move. Fixed-point formatting
        # would print 0.00000 for both all the way through a healthy run.
        log.info(
            "epoch %d | loss %.4f pred %.4f var %.4f cov %.4f margin %.4f "
            "| Var(z) min-dim %.3e | median V(p) %.3e | %.3f s/step (%.1f img/s)",
            epoch, record["loss_loss"], record["loss_pred"], record["loss_var"],
            record["loss_cov"], record["loss_margin"], record["var_z_min_dim"],
            record["v_p_median"], record["sec_per_step"], record["images_per_sec"],
        )

        if v_first_epoch is None and not math.isnan(record["v_p_median"]):
            v_first_epoch = record["v_p_median"]

        save_checkpoint(
            run.checkpoints / "last.pt", model, optimizer, cfg,
            epoch=epoch, step=global_step, best=best_loss,
        )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(
                run.checkpoints / "best.pt", model, optimizer, cfg,
                epoch=epoch, step=global_step, best=best_loss,
            )
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(
                run.checkpoints / f"epoch_{epoch:03d}.pt", model, optimizer, cfg,
                epoch=epoch, step=global_step, best=best_loss,
            )

        _write_figures(run, v_spatial_sum, v_spatial_count, n_tokens)
        _check_v_flat(cfg, record, epoch, v_first_epoch, log)

    log.info("best epoch loss %.4f; checkpoints in %s", best_loss, run.checkpoints)
    print(f"RUN_DIR={run.dir}")


def _accumulate_age_sensitivity(
    out, masks, batch, band_of_row, v_medians, v_by_band, spatial_sum, spatial_count
) -> None:
    """Collect V(p) as a median, per age band, and as a spatial map."""
    v_patch = out.v_patch.detach().float().cpu().numpy()
    idx = masks["tgt_idx"][:, out.margin_block].detach().cpu().numpy()
    keep = masks["tgt_keep"][:, out.margin_block].detach().cpu().numpy()

    for b in range(v_patch.shape[0]):
        values = v_patch[b][keep[b]]
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        v_medians.append(float(np.median(values)))
        band = band_of_row.get(batch["stem"][b])
        if band is not None:
            v_by_band.setdefault(band, []).append(float(np.median(values)))
        positions = idx[b][keep[b]]
        np.add.at(spatial_sum, positions, v_patch[b][keep[b]])
        np.add.at(spatial_count, positions, 1)


def _write_figures(run, spatial_sum, spatial_count, n_tokens: int) -> None:
    history = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    plot_training_curves(history, run.figures / "training_curves.png")
    if any(record.get("v_p_by_band") for record in history):
        plot_band_curves(history, run.figures / "v_p_by_band.png")
    if spatial_count.sum() > 0:
        grid = int(round(n_tokens**0.5))
        mean_v = np.divide(
            spatial_sum, spatial_count,
            out=np.full(n_tokens, np.nan), where=spatial_count > 0,
        ).reshape(grid, grid)
        np.save(run.figures / "v_p_spatial.npy", mean_v)
        plot_spatial_map(
            mean_v, run.figures / "v_p_spatial.png",
            "mean V(p) per patch position (last epoch)",
        )


def _check_v_flat(cfg, record, epoch: int, v_first: float | None, log) -> None:
    """The pre-registered abort: raise lambda_margin and restart, do not hope."""
    abort_epoch = int(cfg.monitor.abort_if_V_flat_after_epoch)
    # epoch 0 supplies the reference, so it cannot also be the epoch under test.
    if epoch == 0 or epoch + 1 < abort_epoch or v_first is None:
        return
    factor = float(cfg.monitor.v_flat_growth_factor)
    current = record["v_p_median"]
    if math.isnan(current) or current <= max(v_first, 1e-12) * factor:
        raise VFlatAbort(
            f"median V(p) is {current:.3e} after {epoch + 1} epochs against "
            f"{v_first:.3e} at the first epoch, short of the {factor:g}x the "
            "pre-registered criterion requires. The predictor is ignoring the age "
            "channel. Raise loss.lambda_margin and restart rather than continuing."
        )
    log.info(
        "age sensitivity check passed: median V(p) %.3e vs %.3e at the first epoch",
        current, v_first,
    )


if __name__ == "__main__":
    main()
