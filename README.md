# Physis / OsteoJEPA

Age-conditioned predictive representations for pediatric wrist fracture triage.

Physis orders the radiologist's reading queue by risk. It does not diagnose, and
it shows the on-call physician nothing before they decide. The product claim
follows the computer-aided triage and notification class (21 CFR 892.2080).

## The idea

Reading pediatric radiographs is hard because normal anatomy is a moving target.
A growth plate looks like a fracture line, and its position and thickness shift
with age: a toddler's carpal bones are still cartilage and invisible, a late
teenager's plates have closed. An image that is normal for a four-year-old is
alarming from a fifteen-year-old. Models that treat "normal" as a single class
lose the variable that separates developmental variation from pathology.

OsteoJEPA conditions on age in the **predictor**, not the encoder, so age can be
intervened on at inference time. For each patch we sweep candidate ages and ask
how much the freedom to pick an age helps explain what we see. Normal anatomy
that looks unusual has some other age that explains it, so its score drops. A
fracture has no age that explains it, so its score stays high. That difference is
the **age-attribution gap**.

The model learns from normal images with no fracture labels, so moving it to a
new population requires only images every hospital already has.

## Status

Under active development for Datathon 2026 (RISTEK Fasilkom UI), semifinal round.

## Setup

Put `physis_meta.zip` and `physis_images_*.zip` in `dataset/`, then:

```bash
bash scripts/setup_env.sh          # CUDA 12.8 build, for RTX 5080 / 5090
bash scripts/setup_env.sh --cpu    # CPU build, enough for the smoke path
```

This creates `.venv`, installs pinned dependencies, extracts the archives into
`data/`, and runs the test suite. Image archives are hosted on Hugging Face; see
`.agents/DATA.md`.

## Running

```bash
bash scripts/run_e1.sh --smoke     # verify the whole pipeline in minutes
.venv/bin/python scripts/check_data.py   # geometry, split, patch label checks
bash scripts/run_base.sh --bench   # 50-step benchmark: throughput and VRAM
bash scripts/run_base.sh           # Stage A pretraining
bash scripts/run_e1.sh             # age mechanism
bash scripts/run_e1b.sh            # architecture ablation
bash scripts/run_e2.sh             # leave-age-band-out
bash scripts/run_e3.sh             # deployment feasibility, no GPU
```

Stage A checkpoints every epoch and can be resumed:

```bash
bash scripts/run_base.sh --resume runs/base/checkpoints/last.pt
```

## Documentation

| File | Contents |
|---|---|
| `CLAUDE.md` | orientation and hard rules |
| `.agents/SPEC.md` | architecture, tensor shapes, losses, inference, pitfalls |
| `.agents/DATA.md` | dataset, manifest schema, filtering, age bands |
| `.agents/EXPERIMENTS.md` | what each experiment tests |
| `.agents/MILESTONES.md` | ordered plan with acceptance criteria |
| `.agents/CONVENTIONS.md` | layout, config, logging, determinism |

## Data

GRAZPEDWRI-DX (Nagy et al., 2022), CC0 Public Domain. 20,327 radiographs from
6,091 patients, University Hospital Graz, 2008–2018.

## License

See `LICENSE`.
