# Research Roadmap

## Completed
- [x] **Week 1** — Spectral compressibility (`NER`, `r90/r95/r99`) & head-rank Spearman agreement across `en`, `es`, `fi`, `qu` on `Qwen3-1.7B`.
- [x] **Week 1b** — Geometric subspace overlap & energy transfer efficiency (`head` vs `joint`).
- [x] **Week 2** — Held-out article-ID splits, out-of-sample noise floor, forward-pass projection, paired bootstrap CIs.
- [x] **Week 3** — Multilingual calibration (`multi` vs `multi_loo`) & `K` vs `V` comparison.
- [x] **Week 4** — Low-budget mixture sweep (`5k / 20k / 150k` tokens, weights `0.00..1.00`) discovering the `5k`-token + `joint` recipe.
- [x] **Week 5** — Position-0 attention-sink control (solved the `K` blow-up & all-layer collapse) + FLORES-200 parallel-corpus control (`flores_to_flores` and `wiki_to_flores`).
- [x] **Week 5b** — Deep dive on `K`: proved standard full-sequence SVD still breaks `K[0]`, quantified `k_norm` angular distortion (`cos_sim 0.689` on pos 0), showed `post-k_norm` cuts blow-up >50%, proved `K` is language-agnostic across all depth quartiles while `V` carries all cross-lingual bias, and demonstrated 4× full-model `K+V` (`layers 3–27`) with `joint` + `protect_pos0` + `mix_5k` (`qu: +0.51%`, `fi: +7.27%`, `es: +14.26%` vs `+428%` English-calibrated).
- [x] **Week 6** — Cross-architecture replication on **`Llama-3.2-3B`** (no QK-Norm): isolated QK-Norm catastrophe (`+153–328%` Qwen3 vs `+5–11%` Llama-3.2 under std SVD; `protect_pos0` still cuts `qu` `joint` K error 5.5×), replicated K-vs-V separation across all depth quartiles, replicated full-model 4× `K+V` (`layers 3–27`: `es +525%`, `fi +266%`, `qu +140%` English-calibrated → rescued by `mix_5k`), and replicated parallel FLORES-200 (`qu` `+0.348` → `+0.002`, `fi` `+0.444` → `+0.009` with 5k tokens!).
- [x] **Week 7** — Downstream evaluation beyond perplexity on **both Qwen3-1.7B & Llama-3.2-3B** (`SIB-200` 7-class topic classification + `FLORES-200` autoregressive MT with real `use_cache=True` compressed KV cache): uncovered literal language-collapse into English under English-only 4× `KV_full_joint` (`qu→es` crashes to `13.3–14.0` chrF++), and showed `mix_5k` beats *every* monolingual basis on bilingual downstream tasks (`qu→es` `+9.6/+10.1` chrF++, Llama-3.2 SIB-200 `qu` `35.3% → 52.0%`).

## Next Steps

### Paper Drafting Checklist
- [ ] Consolidate all CSVs from Kaggle outputs into `results/tables/`.
- [ ] Verify exact definitions against KV-CoRE / Palu / EigenAttention / StreamingLLM baselines.
- [ ] Draft paper sections: Introduction, Method (Gram SVD + `joint` vs `head` + `protect_pos0`), Spectral & Parallel Results, `K` Sink Analysis, Cross-Model Replication, Practical Recommendation (`joint` + `protect_pos0` + `5k` target tokens).
