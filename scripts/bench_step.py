"""M2 verification: ImageNet loading, loss wiring, V(p), and per-step cost.

Four things are checked here, all of which fail silently in a training run:

1. ImageNet weights land on the 1-channel 384px model by *summing* the three
   input channels, not by slicing one of them, and pos_embed is interpolated
   from 14x14 to 24x24.
2. One step of the real ViT-S produces a finite loss.
3. Switching the margin loss off changes the total by exactly
   lambda_margin * L_margin on the same batch, same masks, same weights - which
   is what proves the term is wired in rather than merely computed.
4. Per-step time with and without the margin. The margin adds |D| predictor
   passes and reuses z_ctx and z, so the increase should be around a quarter.
   Close to double means an encoder is being re-invoked for distractor ages.

    .venv/Scripts/python.exe scripts/bench_step.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physis.data.dataset import PhysisDataset, load_manifest, select_split, subset_by_study
from physis.data.masking import build_batch_masks
from physis.losses.margin import age_grid
from physis.losses.objective import compute_objective
from physis.models.osteojepa import OsteoJEPA, momentum_at
from physis.utils.config import load_config
from physis.utils.run import resolve_device, setup_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M2 model and loss checks")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--steps", type=int, default=6, help="timed steps per configuration")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def check_imagenet_load(model: OsteoJEPA, cfg, log) -> dict:
    """Compare the loaded encoder against the original timm checkpoint.

    The model is already on the accelerator by this point while the reference is
    freshly built on the CPU, so everything is compared on the CPU. Both tensors
    have to be on one device for `allclose`, and moving the small reference is
    cheaper than moving the model back.
    """
    source = timm.create_model(str(cfg.encoder.timm_name), pretrained=True, num_classes=0)
    original = source.state_dict()["patch_embed.proj.weight"].cpu()
    loaded = model.encoder.net.patch_embed.proj.weight.detach().cpu()

    summed = original.sum(dim=1, keepdim=True)
    sliced = original[:, :1]
    assert torch.allclose(loaded, summed, atol=1e-6), "patch_embed was not summed across channels"
    slice_gap = float((loaded - sliced).abs().max())
    assert slice_gap > 1e-3, "sum and slice are indistinguishable here; the check proves nothing"

    pos = model.encoder.net.pos_embed.detach().cpu()
    assert pos.shape == (1, 576, model.encoder.embed_dim), f"pos_embed shape {tuple(pos.shape)}"
    assert float(pos.abs().sum()) > 0, "pos_embed is all zeros; interpolation dropped the weights"

    # A block that was actually loaded, versus a freshly initialised predictor.
    block_weight = model.encoder.net.blocks[0].attn.qkv.weight.detach().cpu()
    source_block = source.state_dict()["blocks.0.attn.qkv.weight"].cpu()
    assert torch.allclose(block_weight, source_block, atol=1e-6), "block 0 was not loaded"

    log.info(
        "ImageNet load: patch_embed summed (energy %.4f vs %.4f if sliced), "
        "pos_embed %s, block 0 matches source",
        float(loaded.abs().sum()), float(sliced.abs().sum()), tuple(pos.shape),
    )
    return {
        "patch_embed_summed": True,
        "patch_embed_abs_sum_loaded": float(loaded.abs().sum()),
        "patch_embed_abs_sum_if_sliced": float(sliced.abs().sum()),
        "max_abs_diff_sum_vs_slice": slice_gap,
        "pos_embed_shape": list(pos.shape),
    }


def main() -> None:
    args = parse_args()
    overrides = list(args.set)
    if not any(o.startswith("run.name=") for o in overrides):
        overrides.append("run.name=bench_step")
    if args.batch_size:
        overrides.append(f"optim.batch_size={args.batch_size}")
    cfg = load_config(args.config, overrides)
    run = setup_run(cfg)
    log = run.log

    device = resolve_device(str(cfg.optim.device))
    use_amp = device.type == "cuda" and str(cfg.run.amp) == "bf16"
    batch_size = int(cfg.optim.batch_size)
    log.info(
        "device %s | encoder depth %s | predictor depth %s | batch %d | amp %s",
        device, cfg.encoder.depth, cfg.predictor.depth, batch_size, use_amp,
    )

    manifest = load_manifest(cfg)
    frame = select_split(manifest, "train", clean_only=True)
    frame = subset_by_study(frame, batch_size * (args.steps + args.warmup + 2), int(cfg.run.seed))
    dataset = PhysisDataset(frame, cfg, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model = OsteoJEPA(cfg).to(device)
    imagenet_report = check_imagenet_load(model, cfg, log)

    trainable = (
        list(model.encoder.parameters())
        + list(model.predictor.parameters())
        + list(model.condition.parameters())
    )
    log.info(
        "parameters: encoder %.2fM, predictor %.2fM, condition %.3fM",
        sum(p.numel() for p in model.encoder.parameters()) / 1e6,
        sum(p.numel() for p in model.predictor.parameters()) / 1e6,
        sum(p.numel() for p in model.condition.parameters()) / 1e6,
    )

    grid = age_grid(
        float(cfg.inference.age_min), float(cfg.inference.age_max),
        float(cfg.inference.age_grid_fine_step),
    )

    def to_device(batch):
        return (
            batch["image"].to(device),
            batch["valid_mask"].to(device),
            batch["age"].to(device),
            {k: batch[k].to(device) for k in ("gender", "view", "laterality")},
        )

    # --- the margin term is wired in, on one batch with everything else fixed --
    first = next(iter(loader))
    images, valid, ages, meta = to_device(first)
    masks = build_batch_masks(valid, cfg, np.random.default_rng(0))

    with torch.no_grad():
        with_margin = compute_objective(
            model, images=images, ages=ages, ages_cpu=first["age"].numpy(), meta=meta,
            masks=masks, cfg=cfg, rng=np.random.default_rng(7), grid=grid,
            use_margin=True, margin_block=0,
        )
        without_margin = compute_objective(
            model, images=images, ages=ages, ages_cpu=first["age"].numpy(), meta=meta,
            masks=masks, cfg=cfg, rng=np.random.default_rng(7), grid=grid,
            use_margin=False, margin_block=0,
        )

    lambda_margin = float(cfg.loss.lambda_margin)
    delta_total = float(with_margin.total) - float(without_margin.total)
    expected = lambda_margin * float(with_margin.margin)
    assert torch.isfinite(with_margin.total), "loss is not finite"
    assert abs(delta_total) > 1e-9, "the margin term does not change the total loss"
    assert abs(delta_total - expected) < 1e-5, (
        f"total moved by {delta_total} but lambda_margin * L_margin is {expected}"
    )
    assert float(without_margin.pred) == float(with_margin.pred), (
        "L_pred changed when the margin was switched off; the two paths differ elsewhere"
    )
    log.info(
        "margin wiring: total %.6f -> %.6f (delta %.6f = lambda_m * L_margin %.6f), "
        "predictor calls %d -> %d",
        float(without_margin.total), float(with_margin.total), delta_total, expected,
        without_margin.predictor_calls, with_margin.predictor_calls,
    )

    # V(p) at step 0 is exactly zero, and that is the design rather than a bug:
    # the FiLM MLP is zero-initialized so gamma = 1 and beta = 0 for every c,
    # which makes the residual identical at every age. That point *is* the
    # degenerate solution, which is why the margin loss runs from step 0 with no
    # warmup. The number that matters is V(p) after the margin has had steps to
    # act, measured further down.
    v_init = float(torch.nanmedian(with_margin.v_patch))
    log.info("V(p) at init: computable, median %.3e (zero by construction)", v_init)

    # --- per-step timing ------------------------------------------------------
    def timed(use_margin: bool):
        torch.manual_seed(int(cfg.run.seed))
        local = OsteoJEPA(cfg).to(device)
        local_trainable = (
            list(local.encoder.parameters())
            + list(local.predictor.parameters())
            + list(local.condition.parameters())
        )
        optimizer = torch.optim.AdamW(local_trainable, lr=float(cfg.optim.lr))
        rng = np.random.default_rng(int(cfg.run.seed))
        times: list[float] = []
        iterator = iter(loader)
        for step in range(args.warmup + args.steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            imgs, val, age_t, meta_t = to_device(batch)
            step_masks = build_batch_masks(val, cfg, rng)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = compute_objective(
                    local, images=imgs, ages=age_t, ages_cpu=batch["age"].numpy(),
                    meta=meta_t, masks=step_masks, cfg=cfg, rng=rng, grid=grid,
                    use_margin=use_margin,
                )
            optimizer.zero_grad(set_to_none=True)
            out.total.backward()
            # `local`, not the outer model: clipping the outer parameters would
            # touch tensors that never received a gradient here, so the timed
            # step would silently skip work the real training loop does.
            torch.nn.utils.clip_grad_norm_(local_trainable, float(cfg.optim.grad_clip))
            optimizer.step()
            local.ema_update(momentum_at(step, 100, 0.996, 1.0))
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            assert torch.isfinite(out.total), f"non-finite loss at step {step}"
            if step >= args.warmup:
                times.append(elapsed)
        return float(np.median(times)), local

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    without_time, _ = timed(False)
    with_time, trained = timed(True)
    ratio = with_time / without_time
    log.info(
        "per-step time: %.3f s without margin, %.3f s with margin, ratio %.2fx (+%.0f%%)",
        without_time, with_time, ratio, (ratio - 1) * 100,
    )

    # --- throughput and how long the real run would take -----------------------
    images_per_sec = batch_size / with_time
    steps_per_epoch = int(cfg.clean_set.n_train) // batch_size
    total_steps = steps_per_epoch * int(cfg.optim.epochs)
    projected_hours = total_steps * with_time / 3600.0
    peak_gib = (
        torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
    )
    log.info(
        "throughput %.1f img/s | %d clean train images -> %d steps/epoch, %d steps for %d epochs",
        images_per_sec, int(cfg.clean_set.n_train), steps_per_epoch,
        total_steps, int(cfg.optim.epochs),
    )
    log.info("projected Stage A wall clock: %.1f hours at batch %d", projected_hours, batch_size)
    if device.type == "cuda":
        log.info("peak CUDA memory: %.2f GiB", peak_gib)
    else:
        log.warning(
            "measured on CPU: the projection above is not usable for planning. "
            "Rerun this on the GPU box before committing to the full run."
        )
    if ratio > 1.7:
        log.warning(
            "per-step time nearly doubled. The margin should add predictor passes only; "
            "check that no encoder runs again for distractor ages."
        )

    # --- V(p) after the margin has had steps to act ---------------------------
    n_steps = args.warmup + args.steps
    with torch.no_grad():
        after = compute_objective(
            trained, images=images, ages=ages, ages_cpu=first["age"].numpy(), meta=meta,
            masks=masks, cfg=cfg, rng=np.random.default_rng(7), grid=grid,
            use_margin=True, margin_block=0,
        )
    v_median = float(torch.nanmedian(after.v_patch))
    v_max = float(torch.nan_to_num(after.v_patch, nan=0.0).max())
    log.info(
        "V(p) after %d steps: median %.3e, max %.3e (was exactly 0 at init)",
        n_steps, v_median, v_max,
    )
    assert v_median > 0.0, (
        f"V(p) is still exactly zero after {n_steps} steps: the predictor is ignoring "
        "the age channel and every Stage B quantity would be meaningless"
    )

    run.write_json(
        "bench_step.json",
        {
            "device": str(device),
            "batch_size": batch_size,
            "encoder_depth": int(cfg.encoder.depth),
            "predictor_depth": int(cfg.predictor.depth),
            "imagenet": imagenet_report,
            "loss_with_margin": float(with_margin.total),
            "loss_without_margin": float(without_margin.total),
            "loss_margin_term": float(with_margin.margin),
            "delta_total": delta_total,
            "predictor_calls_with": with_margin.predictor_calls,
            "predictor_calls_without": without_margin.predictor_calls,
            "v_p_median_at_init": v_init,
            "v_p_median_after_steps": v_median,
            "v_p_max_after_steps": v_max,
            "v_p_steps": n_steps,
            "sec_per_step_without_margin": without_time,
            "sec_per_step_with_margin": with_time,
            "margin_time_ratio": ratio,
            "images_per_sec": images_per_sec,
            "steps_per_epoch_full_run": steps_per_epoch,
            "total_steps_full_run": total_steps,
            "projected_hours_full_run": projected_hours,
            "peak_cuda_gib": peak_gib,
        },
    )
    log.info("all M2 checks passed")


if __name__ == "__main__":
    main()
