# Quantitative Results Log (Weeks 1–5)

Model: **`Qwen/Qwen3-1.7B`** (`28` layers, `16` query heads, `8` KV heads, `head_dim = 128`, `D_KV = 1024`, late layers = `21–27`).  
Languages: English (`en` / `eng_Latn`), Spanish (`es` / `spa_Latn`), Finnish (`fi` / `fin_Latn`), Ayacucho Quechua (`qu` / `quy_Latn`).  
All paired confidence intervals are 95% bootstrap CIs (`N_BOOT = 2000`) over evaluation windows (`SEQ_LEN = 512`).

---

## Week 1 — Spectral Structure & Head-Level Compressibility (`01_week1_spectra.ipynb`)

Wikipedia `20231101`, `200,000` tokens/language (`390` windows × `512` tokens, `SKIP_FIRST_TOKENS = 1`).

| Lang | Articles for 200k tok | Baseline PPL | K NER | K `r90` | K `r95` | K `r99` | V NER | V `r90` | V `r95` | V `r99` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `en` | 551 | 15.88 | 0.642 | 0.246 | 0.420 | 0.741 | 0.907 | 0.701 | 0.830 | 0.961 |
| `es` | 477 | 13.21 | 0.643 | 0.248 | 0.421 | 0.741 | 0.902 | 0.691 | 0.824 | 0.959 |
| `fi` | 497 | 15.47 | 0.637 | 0.240 | 0.408 | 0.732 | 0.881 | 0.650 | 0.796 | 0.951 |
| `qu` | 964 | 37.15 | 0.611 | 0.207 | 0.368 | 0.705 | 0.863 | 0.617 | 0.771 | 0.944 |

- **Cross-language head-rank agreement (within-layer Spearman):**
  - `K`: `0.958–0.986` (`en-qu`: `0.958`, `en-fi`: `0.960`, `en-es`: `0.986`).
  - `V`: `0.920–0.971` (`en-qu`: `0.929`, `en-fi`: `0.920`, `en-es`: `0.971`).

---

## Week 1b — Subspace Transfer Efficiency (`02_week1b_subspace_transfer.ipynb`)

Mean energy transfer efficiency $\operatorname{tr}(U_A^\top G_B U_A) / \operatorname{tr}(U_B^\top G_B U_B)$ at `r = 0.25·d` across all 28 layers:

| Kind | Source → Target | Per-Head | Joint Layer |
|---|---|---:|---:|
| `K` | `en → es` | 0.981 | 0.986 |
| `K` | `en → fi` | 0.950 | 0.966 |
| `K` | `en → qu` | 0.963 | 0.980 |
| `V` | `en → es` | 0.905 | 0.940 |
| `V` | `en → fi` | 0.802 | 0.881 |
| `V` | `en → qu` | 0.811 | 0.912 |

---

## Week 2 — Held-Out Splits, Noise Floor & Forward-Pass Intervention (`03_week2_intervention_noise_floor.ipynb`)

Article-ID CRC32 split (`150k` calib / `60k` eval tokens per language).  
Uncompressed baseline PPL on eval split: `en = 16.39`, `es = 12.53`, `fi = 15.89`, `qu = 35.53`.

### Held-out energy efficiency in late layers (`21–27`, `r = 0.25`)

| Kind | Scope | Source | `→ en` | `→ es` | `→ fi` | `→ qu` |
|---|---|---|---:|---:|---:|---:|
| `K` | `head` | `en` | **0.998** | 0.964 | 0.908 | 0.953 |
| `K` | `joint` | `en` | **0.996** | 0.971 | 0.933 | 0.971 |
| `V` | `head` | `en` | **0.975** | 0.825 | 0.659 | 0.734 |
| `V` | `joint` | `en` | **0.980** | 0.899 | 0.831 | 0.904 |

*(Bold diagonal = held-out same-language noise floor; own-language diagonal for `es/fi/qu` on `V` is `0.985–0.994`.)*

### Perplexity increase (%) on `V` (`r = 0.25`, `protect_pos0 = False`)

| Layers | Scope | Target | Matched | `en` basis | `oracle` |
|---|---|---|---:|---:|---:|
| `all` (0–27) | `head` | `en` / `qu` | 1783.7% / 552.7% | 1783.7% / 2739.5% | 1447.4% / 539.5% |
| `all` (0–27) | `joint` | `en` / `qu` | 244.6% / 349.7% | 244.6% / 514.1% | 214.8% / 326.8% |
| `late` (21–27) | `head` | `en` | 11.90% | — | 9.72% |
| `late` (21–27) | `head` | `es` | 12.61% | 14.37% | 11.29% |
| `late` (21–27) | `head` | `fi` | 20.70% | 30.81% | 19.64% |
| `late` (21–27) | `head` | `qu` | 10.08% | **43.49%** | 9.79% |
| `late` (21–27) | `joint` | `en` | 5.24% | — | 3.92% |
| `late` (21–27) | `joint` | `es` | 7.66% | 9.67% | 6.70% |
| `late` (21–27) | `joint` | `fi` | 15.37% | 22.12% | 14.54% |
| `late` (21–27) | `joint` | `qu` | 6.79% | **17.10%** | 5.86% |

---

## Week 3 — Multilingual Calibration `multi` vs `multi_loo` (`04_week3_4_multilingual_and_mixtures.ipynb`)

Late-layer `V`, `r = 0.25`, `ΔNLL` vs matched basis (nats/token, 95% bootstrap CI):

| Scope | Target | `en` basis | `multi_loo` (excl. target) | `multi` (incl. target) | `oracle` |
|---|---|---:|---:|---:|---:|
| `head` | `es` | `+0.0156` `[0.0109, 0.0201]` | `+0.0035` `[-0.0015, 0.0085]` | `-0.0060` `[-0.0100, -0.0019]` | `-0.0118` |
| `head` | `fi` | `+0.0805` `[0.0698, 0.0902]` | `+0.0353` `[0.0263, 0.0445]` | `+0.0011` `[-0.0036, 0.0057]` | `-0.0088` |
| `head` | `qu` | **`+0.2650`** `[0.2315, 0.2978]` | **`+0.2589`** `[0.2271, 0.2905]` | `+0.0200` `[0.0151, 0.0248]` | `-0.0027` |
| `joint` | `es` | `+0.0185` `[0.0150, 0.0222]` | `+0.0028` `[0.0000, 0.0056]` | `+0.0019` `[0.0002, 0.0036]` | `-0.0089` |
| `joint` | `fi` | `+0.0569` `[0.0497, 0.0646]` | `+0.0335` `[0.0267, 0.0407]` | `+0.0098` `[0.0075, 0.0121]` | `-0.0072` |
| `joint` | `qu` | **`+0.0921`** `[0.0818, 0.1031]` | **`+0.0793`** `[0.0695, 0.0884]` | **`-0.0008`** `[-0.0029, 0.0012]` | `-0.0088` |

---

## Week 4 — Low-Budget Target-Language Mixture Sweep (`04_week3_4_multilingual_and_mixtures.ipynb`)

Mixing target-language Gram (`5k`, `20k`, or `150k` tokens) at weight `w` with `multi_loo` (`(1 - w)·multi_loo + w·target`), late-layer `V`, `r = 0.25`, `ΔNLL` vs matched:

| Target | Scope | `w = 0.00` (`multi_loo`) | `w = 0.02` (`5k` / `150k`) | `w = 0.05` (`5k` / `150k`) | `w = 0.10` (`5k` / `150k`) | `w = 0.25` (`5k` / `150k`) |
|---|---|---:|---:|---:|---:|---:|
| `fi` | `head` | 0.035 | 0.011 / 0.011 | 0.006 / 0.008 | 0.001 / 0.003 | -0.001 / 0.001 |
| `fi` | `joint` | 0.034 | 0.009 / 0.010 | 0.007 / 0.008 | 0.007 / 0.009 | 0.009 / 0.010 |
| `qu` | `head` | 0.259 | 0.166 / 0.174 | 0.109 / 0.117 | 0.058 / 0.065 | 0.022 / 0.020 |
| `qu` | `joint` | 0.079 | **0.015** / 0.017 | **0.004** / 0.004 | **0.001** / 0.001 | **0.000** / -0.001 |

---

## Week 5 — Position-0 Sink Control & FLORES-200 Parallel Control (`05_week5_pos0_and_flores.ipynb`)

### Part A — Position-0 Control on Wikipedia (`protect_pos0 = False` vs `True`)

#### A2: `K` (late layers 21–27) — PPL increase (%)
| Scope | `r / d` | Target | Source | `protect_pos0 = False` | `protect_pos0 = True` |
|---|---:|---|---|---:|---:|
| `head` | 0.25 | `en` | `matched` | 175.77% | **1.38%** |
| `head` | 0.25 | `fi` | `matched` / `en` | 337.13% / 372.33% | **1.15%** / **2.23%** |
| `head` | 0.25 | `qu` | `matched` / `en` | 158.65% / 186.76% | **1.99%** / **4.98%** |
| `head` | 0.75 | `en` / `fi` / `qu` | `matched` | 68.83% / 119.62% / 57.28% | **-0.15%** / **0.07%** / **0.25%** |
| `joint` | 0.25 | `en` | `matched` | 47.73% | **0.18%** |
| `joint` | 0.25 | `fi` | `matched` / `en` | 78.54% / 80.97% | **0.41%** / **0.44%** |
| `joint` | 0.25 | `qu` | `matched` / `en` | 37.37% / 43.06% | **0.69%** / **2.18%** |

#### A3: `V` (all layers 0–27, `joint`, `r = 0.25`) — PPL increase (%)
| Target | Source | `protect_pos0 = False` | `protect_pos0 = True` |
|---|---|---:|---:|
| `en` | `matched` | 244.62% | **34.13%** |
| `qu` | `matched` | 349.73% | **7.86%** |
| `qu` | `en` | 513.96% | **70.00%** |

#### A1: `V` (late layers 21–27, `r = 0.25`) — `protect_pos0 = True`
| Scope | Target | Matched PPL (%) | `en` PPL (%) | `ΔNLL` (`en` vs matched) | 95% CI | `ΔNLL` (`multi` vs matched) |
|---|---|---:|---:|---:|---|---:|
| `head` | `fi` | +6.95% | +18.53% | **+0.1028** | `[0.0921, 0.1129]` | +0.0026 (n.s.) |
| `head` | `qu` | +3.74% | +35.46% | **+0.2668** | `[0.2336, 0.2999]` | +0.0216 |
| `joint` | `fi` | +0.49% | +8.06% | **+0.0726** | `[0.0649, 0.0803]` | +0.0017 (n.s.) |
| `joint` | `qu` | -0.89% | +8.00% | **+0.0859** | `[0.0761, 0.0952]` | -0.0045 |

---

### Part B — Parallel-Corpus Control on FLORES-200 `devtest` (`protect_pos0 = True`, `V`, late layers, `r = 0.25`)

Tokenizer fertility (tokens/sentence on identical text): `en = 26.81`, `es = 40.81`, `fi = 52.92`, `qu = 55.21`.  
Uncompressed FLORES `devtest` baseline PPL: `en = 22.93`, `es = 14.89`, `fi = 21.48`, `qu = 168.61`.  
Equalized FLORES `dev` calibration budget: `53` windows (`27,136` tokens/lang).

#### `flores_to_flores` (strictly parallel calibration & evaluation) — `ΔNLL` vs matched (PPL %)
| Scope | Target | Matched PPL | `en` basis | `multi_loo` | `mix_5k` (`w=0.10`) | `multi` | `oracle` |
|---|---|---:|---:|---:|---:|---:|---:|
| `head` | `es` | +2.42% | `+0.0490` (+7.6%) | `+0.0119` (+3.7%) | `+0.0077` (+3.2%) | `+0.0061` (+3.0%) | `-0.0055` |
| `head` | `fi` | +3.85% | `+0.1407` (+19.5%) | `+0.0747` (+11.9%) | `+0.0123` (+5.1%) | `+0.0060` (+4.5%) | `-0.0027` |
| `head` | `qu` | -3.98% | **`+0.6007`** (+75.1%) | **`+0.5562`** (+67.5%) | `+0.0366` (-0.4%) | `-0.0054` (-4.5%) | `+0.0003` |
| `joint` | `es` | -0.54% | `+0.0511` (+4.7%) | `+0.0164` (+1.1%) | `+0.0051` (-0.0%) | `+0.0033` (-0.2%) | `-0.0037` |
| `joint` | `fi` | +0.22% | `+0.1068` (+11.5%) | `+0.0652` (+7.0%) | **`-0.0003`** (+0.2%) | `+0.0001` (+0.2%) | `-0.0033` |
| `joint` | `qu` | -2.17% | **`+0.1468`** (+13.3%) | **`+0.1493`** (+13.6%) | **`-0.0296`** (-5.0%) | **`-0.0256`** (-4.6%) | `+0.0020` |

#### `wiki_to_flores` (Wikipedia bases → parallel FLORES `devtest`, cross-domain transfer) — `ΔNLL` vs matched (PPL %)
| Scope | Target | Matched PPL | `en` basis | `multi_loo` | `mix_5k` (`w=0.10`) | `multi` | `oracle` |
|---|---|---:|---:|---:|---:|---:|---:|
| `head` | `es` | +4.31% | `+0.0227` (+6.7%) | `+0.0097` (+5.3%) | `+0.0062` (+5.0%) | `+0.0028` (+4.6%) | `-0.0238` |
| `head` | `fi` | +6.08% | `+0.0985` (+17.1%) | `+0.0659` (+13.3%) | **`+0.0004`** (+6.1%) | `-0.0002` (+6.1%) | `-0.0240` |
| `head` | `qu` | -2.43% | **`+0.5183`** (+63.8%) | **`+0.5274`** (+65.3%) | `+0.0952` (+7.3%) | `+0.0209` (-0.4%) | `-0.0158` |
| `joint` | `es` | +1.06% | `+0.0343` (+4.6%) | `+0.0189` (+3.0%) | `+0.0094` (+2.0%) | `+0.0058` (+1.7%) | `-0.0196` |
| `joint` | `fi` | +1.42% | `+0.0874` (+10.7%) | `+0.0744` (+9.3%) | `+0.0106` (+2.5%) | `+0.0055` (+2.0%) | `-0.0152` |
| `joint` | `qu` | -1.58% | **`+0.1295`** (+12.0%) | **`+0.1199`** (+11.0%) | **`-0.0086`** (-2.4%) | **`-0.0111`** (-2.7%) | `-0.0039` |


---

## Week 5b — Deep Dive on `K`, `k_norm`, Attention Sinks & Full-Model `K+V` (`06_week5b_k_deepdive.ipynb`)

### Part 1 — Cosine Similarity After `k_norm` (Late Layers, Matched Basis)

| Lang | Projector | `pos 0` (sink) | `pos >= 1` |
|---|---|---:|---:|
| `en` | `head_r25` | **0.6890** | 0.9580 |
| `en` | `head_r75` | **0.8048** | 0.9936 |
| `en` | `joint_r25` | **0.8246** | 0.9841 |
| `qu` | `head_r25` | **0.7018** | 0.9598 |
| `qu` | `head_r75` | **0.8150** | 0.9944 |
| `qu` | `joint_r25` | **0.8292** | 0.9873 |

### Part 2 — Four-Way Position-0 Treatment on `K` (Late Layers, Matched Basis) — PPL Increase (%)

| Scope | `r / d` | Target | `skip1_proj_all` | `gram_all_proj_all` (std SVD) | `protect_pos0` | `protect_pos0_3` |
|---|---:|---|---:|---:|---:|---:|
| `head` | 0.25 | `en` | 175.77% | **169.72%** | **1.38%** | 1.32% |
| `head` | 0.25 | `fi` | 337.13% | **327.62%** | **1.15%** | 1.16% |
| `head` | 0.25 | `qu` | 158.65% | **153.15%** | **1.99%** | 1.97% |
| `head` | 0.75 | `en` / `fi` / `qu` | 68.8% / 119.6% / 57.3% | **53.3% / 88.3% / 41.6%** | **-0.15% / 0.07% / 0.25%** | — |
| `joint` | 0.25 | `en` / `fi` / `qu` | 47.7% / 78.5% / 37.4% | **19.1% / 28.1% / 12.1%** | **0.18% / 0.41% / 0.69%** | 0.15% / 0.38% / 0.70% |

### Part 3 — `pre_knorm` (`k_proj`) vs `post_knorm` (`k_norm`) on `K` (Late Layers, `r = 0.25`) — PPL Increase (%)

| `protect_pos0` | Scope | Target | Source | `pre_knorm` | `post_knorm` |
|---|---|---|---|---:|---:|
| `False` | `head` | `en` / `fi` / `qu` | `matched` | 175.8% / 337.1% / 158.7% | **72.2% / 153.3% / 66.0%** |
| `False` | `joint` | `en` / `fi` / `qu` | `matched` | 47.7% / 78.5% / 37.4% | **36.1% / 63.1% / 26.4%** |
| `True` | `head` | `en` / `fi` / `qu` | `matched` | 1.38% / 1.15% / 1.99% | **-0.37% / -0.31% / 0.72%** |
| `True` | `joint` | `en` / `fi` / `qu` | `matched` | 0.18% / 0.41% / 0.69% | **-0.33% / -0.04% / 0.28%** |
| `True` | `joint` | `fi` / `qu` | `en` (`ΔNLL`) | +0.0002 / +0.0147 | **-0.0022 / +0.0120** |

### Part 4A — Depth-Band Sweep (`joint`, `r = 0.25`, `protect_pos0 = True`) — Foreign-Basis `ΔNLL` (`en` vs Matched)

| Kind | Target | `early` (0–6) | `mid1` (7–13) | `mid2` (14–20) | `late` (21–27) |
|---|---|---:|---:|---:|---:|
| `K` | `fi` | +0.0319 | +0.0030 | +0.0047 | +0.0002 |
| `K` | `qu` | +0.0165 | -0.0048 | -0.0101 | +0.0147 |
| `V` | `fi` | **+0.4456** | +0.0622 | **+0.3968** | +0.0726 |
| `V` | `qu` | **+0.2245** | +0.0355 | +0.0626 | +0.0859 |

### Part 4B — Full-Model 4× `K+V` Compression (`joint`, `r = 0.25`, `protect_pos0 = True`) — PPL Increase (%) & `ΔNLL` vs Matched

#### `layers_3_27` (25 of 28 layers, skipping only the layers 0–2 sink spike)
| Target | Matched PPL | `en` basis | `multi_loo` | `mix_5k` (`w=0.10`) | `multi` |
|---|---:|---:|---:|---:|---:|
| `en` | +12.35% | — | +17.27% (`+0.0428`) | +16.34% (`+0.0349`) | +15.08% (`+0.0239`) |
| `es` | +10.07% | **+427.61%** (`+1.5673`) | +17.36% (`+0.0642`) | **+14.26%** (`+0.0374`) | **+11.99%** (`+0.0174`) |
| `fi` | +3.11% | **+260.38%** (`+1.2514`) | +84.69% (`+0.5829`) | **+7.27%** (`+0.0396`) | **+4.79%** (`+0.0163`) |
| `qu` | **-1.32%** | **+25.18%** (`+0.2378`) | +14.97% (`+0.1527`) | **+0.51%** (`+0.0183`) | **-0.98%** (`+0.0034`) |

#### `all` (all 28 layers, 0–27)
| Target | Matched PPL | `en` basis | `multi_loo` | `mix_5k` (`w=0.10`) | `multi` |
|---|---:|---:|---:|---:|---:|
| `en` | +35.38% | — | +189.62% (`+0.7605`) | +86.41% (`+0.3199`) | +59.04% (`+0.1611`) |
| `es` | +26.07% | +174.45% (`+0.7779`) | +59.24% (`+0.2336`) | +44.92% (`+0.1394`) | +34.83% (`+0.0672`) |
| `fi` | +24.77% | +294.38% (`+1.1508`) | +146.80% (`+0.6821`) | **+26.98%** (`+0.0176`) | **+20.52%** (`-0.0347`) |
| `qu` | +9.86% | +77.76% (`+0.4812`) | +48.28% (`+0.2999`) | **+15.23%** (`+0.0477`) | **+10.95%** (`+0.0099`) |


---

## Week 6 — Cross-Architecture Replication on `Llama-3.2-3B` (`07_week6_llama32_replication.ipynb`)

Model: **`unsloth/Llama-3.2-3B`** (`28` layers, `24` query heads, `8` KV heads, `head_dim = 128`, `D_KV = 1024`, **`has_k_norm = False`**).  
Tokenizer fertility on FLORES-200 (tokens/sentence): `en = 26.36`, `es = 40.64`, `fi = 52.32`, `qu = 54.42`.  
Uncompressed baseline PPL — Wikipedia: `en = 9.35`, `es = 7.22`, `fi = 5.40`, `qu = 10.16` | FLORES `devtest`: `en = 19.42`, `es = 11.82`, `fi = 9.42`, `qu = 56.64`.

### Part 1 — Spectral Compressibility & Cosine Similarity (`no k_norm`)

| Lang | K NER | K `r90` | K `r95` | K `r99` | V NER | V `r90` | V `r95` | V `r99` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `en` | 0.587 | 0.183 | 0.347 | 0.660 | 0.931 | 0.751 | 0.863 | 0.970 |
| `es` | 0.580 | 0.176 | 0.334 | 0.650 | 0.919 | 0.729 | 0.849 | 0.967 |
| `fi` | 0.561 | 0.152 | 0.302 | 0.629 | 0.899 | 0.689 | 0.823 | 0.959 |
| `qu` | 0.548 | 0.136 | 0.282 | 0.619 | 0.893 | 0.681 | 0.819 | 0.959 |

- Within-layer Spearman agreement: `K` `0.955–0.985`, `V` `0.781–0.920`.
- Late-layer `K` cosine similarity (`pos 0` vs `pos >= 1`): `head_r25` `0.7515` vs `0.9571`; `joint_r25` `0.8440` vs `0.9848`.

### Part 2 — Position-0 Treatment on `K` Without `k_norm` (Late Layers, `r = 0.25`, Matched) — PPL Increase (%)

| Scope | Target | `skip1_proj_all` | `gram_all_proj_all` (std SVD) | `protect_pos0` |
|---|---|---:|---:|---:|
| `head` | `en` / `fi` / `qu` | 5.12% / 7.14% / 10.72% | 5.10% / 7.09% / 10.63% | **4.40% / 4.43% / 4.69%** |
| `joint` | `en` / `fi` / `qu` | 1.86% / 2.67% / 4.20% | 1.65% / 2.05% / 3.34% | **0.90% / 0.66% / 0.61%** |

### Part 3 — Depth-Band Sweep (`joint`, `r = 0.25`, `protect_pos0 = True`) — `ΔNLL` (`en` vs Matched)

| Kind | Target | `early` (0–6) | `mid1` (7–13) | `mid2` (14–20) | `late` (21–27) |
|---|---|---:|---:|---:|---:|
| `K` | `fi` | +0.0102 | +0.0025 | +0.0087 | +0.0129 |
| `K` | `qu` | +0.0091 | +0.0137 | +0.0066 | +0.0256 |
| `V` | `fi` | **+0.3695** | +0.0355 | +0.1025 | **+0.1552** |
| `V` | `qu` | **+0.1253** | +0.0507 | **+0.2018** | **+0.2345** |

### Part 4 — Full-Model 4× `K+V` (`layers 3–27`, `joint`, `r = 0.25`, `protect_pos0 = True`) — PPL (%) & `ΔNLL` vs Matched

| Target | Matched PPL | `en` basis | `multi_loo` | `mix_5k` (`w=0.10`) | `multi` |
|---|---:|---:|---:|---:|---:|
| `en` | +44.91% | — | +65.03% (`+0.1300`) | +56.70% (`+0.0782`) | +49.90% (`+0.0338`) |
| `es` | +34.26% | **+524.78%** (`+1.5376`) | +41.95% (`+0.0557`) | **+39.25%** (`+0.0365`) | **+35.90%** (`+0.0122`) |
| `fi` | +22.29% | **+266.43%** (`+1.0974`) | +262.15% (`+1.0856`) | **+34.55%** (`+0.0955`) | **+27.69%** (`+0.0432`) |
| `qu` | +14.77% | **+139.92%** (`+0.7374`) | +99.58% (`+0.5533`) | **+31.58%** (`+0.1367`) | **+24.36%** (`+0.0803`) |

### Part 5 — Parallel-Corpus Control on FLORES-200 (`V`, Late Layers, `r = 0.25`, `protect_pos0 = True`) — `ΔNLL` vs Matched (PPL %)

| Regime & Scope | Target | Matched PPL | `en` basis | `multi_loo` | `mix_5k` (`w=0.10`) | `multi` |
|---|---|---:|---:|---:|---:|---:|
| `flores_to_flores` (`head`) | `es` | +3.83% | `+0.2166` (+29.0%) | `+0.0538` (+9.6%) | `+0.0146` (+5.4%) | `+0.0099` (+4.9%) |
| `flores_to_flores` (`head`) | `fi` | +4.27% | **`+0.6014`** (+90.3%) | **`+0.5065`** (+73.0%) | **`+0.0132`** (+5.7%) | `+0.0060` (+4.9%) |
| `flores_to_flores` (`head`) | `qu` | +1.97% | **`+0.5235`** (+72.1%) | **`+0.4873`** (+66.0%) | **`+0.0187`** (+3.9%) | `+0.0068` (+2.7%) |
| `flores_to_flores` (`joint`) | `es` | +1.30% | `+0.1787` (+21.1%) | `+0.0450` (+6.0%) | `+0.0148` (+2.8%) | `+0.0061` (+1.9%) |
| `flores_to_flores` (`joint`) | `fi` | +1.17% | **`+0.4444`** (+57.8%) | **`+0.4632`** (+60.8%) | **`+0.0090`** (+2.1%) | `+0.0037` (+1.6%) |
| `flores_to_flores` (`joint`) | `qu` | -0.14% | **`+0.3481`** (+41.4%) | **`+0.3497`** (+41.7%) | **`+0.0022`** (+0.1%) | **`-0.0003`** (-0.2%) |
| `wiki_to_flores` (`joint`) | `es` | +4.12% | `+0.0504` (+9.5%) | `+0.0100` (+5.2%) | `+0.0090` (+5.1%) | `+0.0066` (+4.8%) |
| `wiki_to_flores` (`joint`) | `fi` | +2.87% | `+0.2062` (+26.4%) | `+0.3955` (+52.8%) | **`+0.0091`** (+3.8%) | `+0.0023` (+3.1%) |
| `wiki_to_flores` (`joint`) | `qu` | +2.48% | **`+0.3560`** (+46.3%) | `+0.2663` (+33.8%) | **`+0.0115`** (+3.7%) | **`+0.0062`** (+3.1%) |


---

## Week 7 — Downstream Evaluation Beyond Perplexity (`08_week7_downstream_eval.ipynb`)

Calibrated **only on Wikipedia** (strict zero-leakage cross-domain evaluation), evaluated on both **`Qwen/Qwen3-1.7B`** and **`unsloth/Llama-3.2-3B`** under `V_late_head` (`layers 21–27`, `V`, `head`, `r = 0.25`) and `KV_full_joint` (`layers 3–27`, `K+V`, `joint`, `r = 0.25`, 4× full-model compression, `protect_pos0 = True`).

### Task 1 — SIB-200 Topic Classification (7 classes, 204 parallel test examples/lang, 7-shot) — Accuracy (%)

| Model | Regime | Target | Uncomp. | Matched | `en` basis | `multi_loo` | **`mix_5k`** | `multi` |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen3-1.7B` | `V_late_head` | `en` / `es` / `fi` / `qu` | 75.0 / 75.5 / 74.0 / 44.6 | 80.4 / 80.9 / 77.9 / 49.0 | — / 79.4 / 76.0 / 47.5 | 82.4 / 80.9 / 77.5 / 51.5 | **81.9 / 81.4 / 78.9 / 51.5** | 82.8 / 81.9 / 77.9 / 51.0 |
| `Qwen3-1.7B` | `KV_full_joint` | `en` / `es` / `fi` / `qu` | 75.0 / 75.5 / 74.0 / 44.6 | 72.5 / 58.3 / 57.4 / 35.8 | — / 70.6 / 60.3 / 38.7 | 58.3 / 64.2 / 60.3 / 37.7 | 66.7 / 64.7 / 60.3 / 37.3 | 65.2 / 65.7 / 57.4 / **40.7** |
| `Llama-3.2-3B` | `V_late_head` | `en` / `es` / `fi` / `qu` | 77.5 / 78.4 / 77.5 / 56.9 | 70.6 / 73.0 / 78.4 / 51.0 | — / 73.5 / **72.1** / 51.5 | 76.5 / 77.0 / 76.5 / 51.5 | **76.5 / 77.0 / 79.9 / 52.5** | 74.5 / 78.4 / 78.9 / 51.5 |
| `Llama-3.2-3B` | `KV_full_joint` | `en` / `es` / `fi` / `qu` | 77.5 / 78.4 / 77.5 / 56.9 | 81.9 / 83.8 / 77.0 / **33.3** | — / 82.4 / 77.5 / **35.3** | 79.4 / 77.5 / 80.4 / 47.5 | **78.4 / 78.4 / 80.4 / 52.0** | 79.9 / 79.4 / 81.4 / **51.0** |

*Key paired bootstrap CIs:*
- `Llama-3.2-3B`, `V_late_head`, `fi`: `en` vs matched = **`-6.37 pp`** `[-10.29, -2.94]` → recovered by `mix_5k` (`79.9%`).
- `Llama-3.2-3B`, `KV_full_joint`, `qu`: both monolingual bases (`en`: `35.3%`, `qu`-only `matched`: `33.3%`) fail because the prompt is bilingual (English instructions/labels + Quechua text); **`mix_5k`** jumps to **`52.0%`** (**`+18.63 pp` over `matched`, `[+13.24, +24.51]`**).

### Task 2 — FLORES-200 Autoregressive Machine Translation (`N = 80`, 5-shot, real `use_cache=True` decoding) — chrF++

| Model | Regime | Direction | Uncomp. | Matched | `en` basis | `multi_loo` | **`mix_5k`** | `multi` |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen3-1.7B` | `V_late_head` | `qu->es` | 27.52 | 26.38 | 27.03 | 26.84 | 26.83 | 26.79 |
| `Qwen3-1.7B` | `V_late_head` | `fi->en` / `es->en` | 50.74 / 57.58 | 47.65 / 56.92 | 47.36 / 57.22 | 48.72 / 57.36 | 48.37 / 57.01 | 47.80 / 57.07 |
| `Qwen3-1.7B` | **`KV_full_joint`** | **`qu->es`** | **27.52** | 21.04 | **13.26** *(-7.54 vs matched)* | 23.19 | **22.85** *(+9.6 over `en`)* | **22.94** |
| `Qwen3-1.7B` | **`KV_full_joint`** | `fi->en` | 50.74 | 32.01 | **28.55** | 32.42 | **32.71** | **32.75** |
| `Qwen3-1.7B` | **`KV_full_joint`** | `es->en` | 57.58 | 50.61 | **41.12** | 49.45 | **48.65** | **49.04** |
| `Llama-3.2-3B` | `V_late_head` | `qu->es` | 27.47 | 26.70 | 27.33 | 26.64 | 27.48 | 27.47 |
| `Llama-3.2-3B` | `V_late_head` | `fi->en` / `es->en` | 55.02 / 59.92 | 52.77 / 58.90 | 52.73 / 59.00 | 52.60 / 59.05 | 52.59 / 58.99 | 51.67 / 58.74 |
| `Llama-3.2-3B` | **`KV_full_joint`** | **`qu->es`** | **27.47** | 20.66 | **14.01** | 23.12 | **24.14** *(+10.1 over `en`)* | **23.46** |
| `Llama-3.2-3B` | **`KV_full_joint`** | **`es->qu`** | **19.98** | 16.59 | **10.26** | 16.19 | **14.90** *(+4.6 over `en`)* | 14.58 |
| `Llama-3.2-3B` | **`KV_full_joint`** | **`fi->en`** | **55.02** | 33.17 | **29.63** | 37.38 | **38.58** *(+9.0 over `en`)* | **38.73** |
| `Llama-3.2-3B` | **`KV_full_joint`** | `es->en` | 59.92 | 42.32 | **42.23** | 46.31 | **46.17** | **47.25** |

### Qualitative Failure Mode under English-Only Calibration (`qu -> es`, `KV_full_joint`)
English-calibrated 4× KV-cache compression causes **literal language-collapse into English** during Quechua→Spanish autoregressive generation in **both** models, whereas `mix_5k` preserves Spanish output and named entities:
- **Qwen3-1.7B (`KV_full | en`):** `"The pilot was the first to be identified in the world, and the first to be identified..."` → **`KV_full | mix_5k`:** `"El Dr. Ehud Ur, el director de la escuela de medicina..."`
- **Llama-3.2-3B (`KV_full | en`):** `"Dr. Ehud Ur, yachachiq of the Dalhosie Suntur, was appointed as the new director..."` → **`KV_full | mix_5k`:** `"Dr. Ehud Ur, el director de la escuela de medicina..."`
