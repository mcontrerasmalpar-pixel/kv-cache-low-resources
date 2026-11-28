# Paper Draft (`paper/`)

This directory contains the complete, Overleaf-ready LaTeX manuscript for:

> **Cross-Lingual Calibration Bias in Low-Rank KV-Cache Compression: Attention Sinks, QK-Norm Amplification, and Low-Resource Recovery**  
> Maria Contreras — Disentangle Lab

## Files
- `main.tex` — Complete 2-column ACL/EMNLP-style manuscript with all 6 quantitative/qualitative tables, 4 multi-panel figures, limitations, and reproducibility statement.
- `references.bib` — Verified BibTeX entries (Palu, Eigen Attention, KV-CoRE, StreamingLLM, Massive Activations, Marchisio et al., Qwen3, Llama 3, FLORES-200, SIB-200, chrF++).

## Compiling on Overleaf
1. Upload `paper/main.tex`, `paper/references.bib`, and the 5 referenced PNG figures from `results/figures/` into a `figures/` folder on Overleaf:
   - `week5b_fig2_k_pos0_treatments_and_knorm.png`
   - `week5b_fig3_layer_bands_k_vs_v.png`
   - `week6_fig1_llama32_pos0_and_depth.png`
   - `week6_fig3_llama32_flores_parallel.png`
   - `week7_fig2_mt_flores_downstream.png`
2. Compile directly with `pdfLaTeX` (standard Overleaf packages only — zero missing `.sty` files).
3. For official ACL/EMNLP/ARR submission, upload `acl.sty` + `acl_natbib.bst`, swap `\usepackage[margin=2.0cm,columnsep=0.65cm]{geometry}` for `\usepackage[review]{acl}`, and switch `\bibliographystyle{plainnat}` to `\bibliographystyle{acl_natbib}`.
