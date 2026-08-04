"""Stage A pretraining.

    L = L_pred + lambda_v * L_var + lambda_c * L_cov + lambda_m * L_margin

Only clean images from the training fold enter here. The model must never see
pathology during Stage A; that condition is what makes the residual meaningful.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physis.data.dataset import PhysisDataset, load_manifest, select_split, subset_by_study
from physis.data.masking import build_batch_masks
from physis.losses.jepa import patch_residual, prediction_loss
from physis.losses.margin import age_grid, age_sensitivity, margin_loss, sample_distractor_ages
from physis.losses.vicreg import covariance_loss, variance_loss
from physis.models.osteojepa import OsteoJEPA, gather_tokens, momentum_at
from physis.utils.config import load_config
from physis.utils.run import resolve_device, setup_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OsteoJEPA Stage A pretraining")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="key.sub=value overrides")
    parser.add_argument("--run-subdir", default=None)
    return parser.parse_args()


def lr_at(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    run = setup_run(cfg, subdir=args.run_subdir)
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

    dataset = PhysisDataset(train_df, cfg, augment=bool(cfg.data.augment), seed=int(cfg.run.seed))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.optim.batch_size),
        shuffle=True,
        num_workers=int(cfg.optim.num_workers),
        drop_last=len(dataset) >= int(cfg.optim.batch_size),
        pin_memory=device.type == "cuda",
    )

    model = OsteoJEPA(cfg).to(device)
    trainable = list(model.encoder.parameters()) + list(model.predictor.parameters()) + list(
        model.condition.parameters()
    )
    optimizer = torch.optim.AdamW(
        trainable, lr=float(cfg.optim.lr), weight_decay=float(cfg.optim.weight_decay)
    )

    steps_per_epoch = max(len(loader), 1)
    epochs = int(cfg.optim.epochs)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * int(cfg.optim.warmup_epochs)
    log.info("steps: %d per epoch, %d total", steps_per_epoch, total_steps)

    grid = age_grid(
        float(cfg.inference.age_min),
        float(cfg.inference.age_max),
        float(cfg.inference.age_grid_fine_step),
    )
    rng = np.random.default_rng(int(cfg.run.seed))
    lambda_margin = float(cfg.loss.lambda_margin)
    global_step = 0

    for epoch in range(epochs):
        model.train()
        totals = {k: 0.0 for k in ("loss", "pred", "var", "cov", "margin")}
        v_medians, std_mins, step_times = [], [], []

        for batch in loader:
            step_start = time.perf_counter()
            lr = lr_at(global_step, total_steps, warmup_steps, float(cfg.optim.lr), float(cfg.optim.min_lr))
            for group in optimizer.param_groups:
                group["lr"] = lr

            images = batch["image"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True)
            age = batch["age"].to(device)
            meta = {k: batch[k].to(device) for k in ("gender", "view", "laterality")}

            masks = build_batch_masks(valid, cfg, rng)
            ctx_idx, ctx_keep = masks["ctx_idx"], masks["ctx_keep"]
            tgt_idx, tgt_keep = masks["tgt_idx"], masks["tgt_keep"]
            n_blocks = tgt_idx.shape[1]
            margin_block = int(rng.integers(n_blocks))

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                z_full = model.encode_targets(images)
                z_ctx = model.encode_context(images, ctx_idx, ctx_keep)
                cond = model.condition_vector(age, meta)

                loss_pred = images.new_zeros(())
                residual_for_margin = None
                z_tgt_for_margin = None
                for b in range(n_blocks):
                    idx_b, keep_b = tgt_idx[:, b], tgt_keep[:, b]
                    z_tgt = gather_tokens(z_full, idx_b)
                    pred = model.predict(z_ctx, ctx_idx, ctx_keep, idx_b, keep_b, cond)
                    residual = patch_residual(pred, z_tgt)
                    loss_pred = loss_pred + prediction_loss(residual, keep_b)
                    if b == margin_block:
                        residual_for_margin = residual
                        z_tgt_for_margin = z_tgt
                loss_pred = loss_pred / n_blocks

                loss_var, std_per_dim = variance_loss(z_ctx, ctx_keep)
                loss_cov = covariance_loss(z_ctx, ctx_keep)

                if lambda_margin > 0:
                    distractors = sample_distractor_ages(
                        batch["age"].numpy(), grid, float(cfg.loss.distractor_tau), rng
                    )
                    distractors_t = torch.from_numpy(distractors).to(device)
                    loss_margin, s_distractor = margin_loss(
                        model,
                        z_ctx=z_ctx,
                        ctx_idx=ctx_idx,
                        ctx_keep=ctx_keep,
                        tgt_idx=tgt_idx[:, margin_block],
                        tgt_keep=tgt_keep[:, margin_block],
                        z_tgt=z_tgt_for_margin,
                        meta=meta,
                        residual_recorded=residual_for_margin,
                        distractor_ages=distractors_t,
                        margin=float(cfg.loss.margin_m),
                    )
                    all_ages = torch.cat(
                        [residual_for_margin.unsqueeze(1), s_distractor], dim=1
                    ).detach()
                    v_patch = age_sensitivity(all_ages, tgt_keep[:, margin_block])
                    v_medians.append(float(torch.nanmedian(v_patch)))
                else:
                    loss_margin = images.new_zeros(())

                loss = (
                    loss_pred
                    + float(cfg.loss.lambda_var) * loss_var
                    + float(cfg.loss.lambda_cov) * loss_cov
                    + lambda_margin * loss_margin
                )

            assert torch.isfinite(loss), f"non-finite loss at step {global_step}"
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(cfg.optim.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(trainable, float(cfg.optim.grad_clip))
            optimizer.step()
            model.ema_update(
                momentum_at(
                    global_step,
                    total_steps,
                    float(cfg.target_encoder.ema_start),
                    float(cfg.target_encoder.ema_end),
                )
            )

            totals["loss"] += float(loss)
            totals["pred"] += float(loss_pred)
            totals["var"] += float(loss_var)
            totals["cov"] += float(loss_cov)
            totals["margin"] += float(loss_margin)
            if std_per_dim.numel():
                std_mins.append(float(std_per_dim.min()))
            step_times.append(time.perf_counter() - step_start)
            global_step += 1

            if global_step % int(cfg.run.log_every) == 0:
                log.info("step %d/%d loss %.4f lr %.2e", global_step, total_steps, float(loss), lr)

        record = {
            "epoch": epoch,
            "lr": lr,
            **{f"loss_{k}": v / steps_per_epoch for k, v in totals.items()},
            # Var(z) is tracked as the *minimum* across dimensions: a collapse
            # confined to a few dimensions never shows up in the mean.
            "var_z_min_dim": float(np.min(std_mins) ** 2) if std_mins else float("nan"),
            "v_p_median": float(np.median(v_medians)) if v_medians else float("nan"),
            "sec_per_step": float(np.mean(step_times)),
        }
        run.append_metrics(record)
        # V(p) and Var(z) start many orders of magnitude below 1, and the whole
        # point of logging them is to watch them move. Fixed-point formatting
        # would print 0.00000 for both all the way through a healthy run.
        log.info(
            "epoch %d | loss %.4f pred %.4f var %.4f cov %.4f margin %.4f "
            "| Var(z) min-dim %.3e | median V(p) %.3e | %.3f s/step",
            epoch,
            record["loss_loss"],
            record["loss_pred"],
            record["loss_var"],
            record["loss_cov"],
            record["loss_margin"],
            record["var_z_min_dim"],
            record["v_p_median"],
            record["sec_per_step"],
        )

    checkpoint = {
        "model": model.state_dict(),
        "config": {"encoder_depth": int(cfg.encoder.depth), "predictor_depth": int(cfg.predictor.depth)},
        "epoch": epochs,
    }
    path = run.checkpoints / "best.pt"
    torch.save(checkpoint, path)
    log.info("saved checkpoint: %s", path)
    print(f"RUN_DIR={run.dir}")


if __name__ == "__main__":
    main()
