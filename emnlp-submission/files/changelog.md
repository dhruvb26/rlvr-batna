# Change Log — `paper.tex` (uncommitted, since last commit)

Baseline: commit `c600bf4` ("rename report to submission folder"). Diff: 117 insertions, 60 deletions.
This log summarizes every change currently in `paper.tex` relative to that commit, grouped by
purpose, for use in an ARR resubmission response / change note.

---

## A. Claim scoping and clarifications (reviewer-driven)

Addresses y3Tq (W1–W3), DPWH (W3, C3), and SmhC (W3).

- **Abstract:** "no natural structural floor" → "no task-given utility floor." Softened/《corrected》
  the results claim from "consistently achieves the highest overall average bargained ratio,
  outperforming … larger frontier models" to "achieves the highest average bargained ratio against
  both the fixed training opponent and a stronger frontier opponent, outperforming … far larger
  frontier models."
- **Introduction:** "surplus rewards that treat any closed deal as positive" →
  "…treat any closed *multi-item* deal as positive (§ Reward Design)." Scopes the claim to
  multi-item settings, conceding the price-side case raised by y3Tq (W3, Liu et al.).
- **Methodology (§2):** clarified that the regulation mechanism "applies only to the frozen
  opponent, and the learner never accesses its counterpart's constraints" (SmhC W3).
- **Regulation asymmetry (§2):** rewrote "it ensures $P \geq \mathcal{C}$" as "it ensures the
  negotiated price $P$ never falls below the seller's private cost $\mathcal{C}$," defining $P$ and
  $\mathcal{C}$ at first use (DPWH C3, "P and C not defined").
- **Surplus reward (§2.1):** added the structural argument that in multi-item allocation item
  values are non-negative, so $\rho \in [0,1]$ for every feasible allocation — a utility floor
  cannot arise from the task and must be imposed by the reward designer (core rebuttal to y3Tq W1,
  "just use negative utilities").
- **Limitations:** added that "the trained policy itself … requires no access to its counterpart's
  constraints at inference time" (SmhC W3).
- Minor grammar: "normalized surplus $\rho$ satisfies" → "satisfying."

## B. Train/test split documentation (DPWH W3/C1)

- **§ Experimental Results:** "We evaluate on all four training sets" → "We evaluate on held-out
  test splits of all four training datasets (Appendix A)…".
- **Appendix A (Training Details):** added a new **"Train/test splits"** paragraph documenting the
  disjoint pools: CaSiNo 906/124, DnD 5,048/526, CraigslistBargains 4,020/637, AHP product-level
  743/187 products (422/93 usable, stratified by category), JI fully held out (295).
- **Open item flagged in-source:** a `% TODO(rebuttal)` comment notes that the AHP numbers in the
  results tables must be re-run on the disjoint `data/ahp/test` split before this ships. (The
  code/data split now exists; the paper's AHP table values are not yet regenerated on it.)

## C. Metric reporting: per-episode clipping + recomputed tables

- **§ Experimental Results:** added "Consistent with the training reward, we clip the bargained
  ratio to $[-1,1]$ when reporting, so that a handful of degenerate thin-margin price episodes
  cannot dominate the per-dataset mean." (Addresses DPWH's small-eval-set concern about outlier
  sensitivity.)
- **Recomputed bargained-ratio tables** with per-episode clipping applied before averaging. Notable
  cell changes (Table vs. Qwen3-30B-A3B): Surplus AHP .47→.64; Qwen3-30B AHP −.56→−.05, CRA
  −.02→.01; Qwen3.6-35B AHP .37→.46; Llama-4-Maverick AHP −1.61→.07, CRA −.19→−.08; plus smaller
  updates to Qwen3-235B, GPT-5.4-mini, DeepSeek, Kimi. Table vs. GPT-5.4 similarly recomputed
  (e.g., Threshold AHP .01→.35; Surplus AHP −.02→.18; Llama AHP −1.61→−.14).
- **Narrative numbers updated to match:** surplus average .57→.60; price-task surplus .54→.62;
  AHP recovery "$-0.56$ (base) → 0.80" → "$-0.05$ (base) → 0.80"; GPT-5.4-opponent appendix text
  updated (threshold AHP drop ".80 → .01" → ".80 → .35"; added the clipped five-benchmark average
  0.51 vs. 0.45; DnD Pareto 0.67 vs. 0.58 phrasing).
- **First-bid paragraph** rewritten to give explicit evaluation numbers (0.36/0.40 on AHP/CRA vs.
  0.59/0.56 surplus and 0.95/0.92 base) instead of only the ≈30%/45%/89% training-curve values.
- **Cross-task transfer paragraph** rephrased ("comparable to frontier models 8–200× larger, most
  spanning 0.71–0.79 on JI, and well above surplus (0.63)").

## D. New appendix: Evaluation Metrics

- Added **Appendix B "Evaluation Metrics"** (`app:metrics`) with formal definitions of the
  first-bid ratio (Eq. `first_bid`) and Pareto efficiency rate (Eq. `pareto`), referenced from the
  auxiliary-metric tables.

## E. Transfer ablation: hypothesis resolved (SmhC C1, DPWH W2)

- **§3 "Aggressive anchoring" paragraph rewritten:** the previously open hypothesis ("walk-away
  tolerance learned on multi-item scenarios transfers to price bargaining … isolating the mechanism
  would require training on multi-item tasks alone") is replaced with the resolved finding: a
  controlled multi-item-only ablation shows **no positive transfer to price**, so the joint agent's
  price gains come from the direct price-task surplus reward, not cross-task transfer (points to
  Appendix `app:transfer`).
- **New Appendix "Transfer Ablation: Multi-Item-Only Training"** (`app:transfer`, Table
  `tab:transfer`): threshold agent trained on CaSiNo+DnD only, evaluated on all five datasets
  against the same-base untrained model (100 episodes/cell). Strong on trained tasks (DnD 0.70 vs.
  0.53 base), no transfer on held-out price (AHP 0.13 vs. 0.32; first-bid 0.84/0.89 vs. 0.80/0.82)
  or JI (0.70 vs. 0.74). Includes the format-discipline side finding (base violates the
  Thought/Talk/Action format on 55–63% of multi-item/JI turns; trained agent <3%).
- **Base-model disclosure included in the appendix:** the original base
  (Qwen3-30B-A3B-Instruct-2507) was retired by the training provider after submission; the ablation
  uses the official replacement (Qwen3.6-35B-A3B) with identical hyperparameters, disjoint AHP
  split, the same evaluation opponent, and same-base baselines.
- **Contribution (3) softened:** "strong transfer to unseen negotiation structures" → "transfer to
  unseen negotiation structures" (consistency with the now-resolved transfer question).

## F. Why the threshold is applied selectively (DPWH C2)

- **New Appendix "Why the Threshold is Applied Selectively"** (`app:uniform`): a post-hoc analysis
  motivating (not proving) the selective design. Corrected regulation wording: it only prevents
  sub-cost prices (P ≥ C, non-negative *seller* surplus), so the *buyer's* surplus can be thin.
  Reports counts with denominators for positive-surplus price deals below τ=0.4: 1/73 (AHP) and
  11/78 (CRA) vs. the training opponent, 23/66 and 33/74 vs. GPT-5.4, and 15/23 and 23/26 for the
  untrained base — the share grows with opponent strength, and since the best agent reaches only
  0.32–0.35 BR on price vs. GPT-5.4 (Appendix `app:frontier`), many such deals are legitimately
  thin-margin. Explicitly framed as descriptive, single-checkpoint reclassification; a controlled
  uniform-threshold training run is deferred to future work. (An earlier draft overclaimed this as
  causal evidence of "harm"/"deadlock"; the reframed version states it as motivation only.)

## G. Code availability (y3Tq/DPWH Datasets & Software = 1)

- **Added a footnote to the contributions paragraph** stating that all training/evaluation code,
  configs, and data splits will be released; a `% TODO(resubmission)` marker reserves the spot for
  the anonymized repository URL once created.

## H. Formatting and figure captions

- **Split the wide deal-closure table** (formerly one 11-column, two-panel table `tab:deal_rates`)
  into two single-panel tables: `tab:deal_rates_qwen` (vs. Qwen3-30B-A3B) and `tab:deal_rates_gpt`
  (vs. GPT-5.4); updated the referring sentence in § Outcome Breakdown and the summary range
  ("76–92%" → "76–87%"). Column spacing widened (`tabcolsep` 3pt→4pt).
- **Figure captions expanded:** reward-curves caption now explains the 5-step moving average and
  raw-reward bands; per-dataset price-curve caption updated ("reaching ~0.8 by step 60" → "peaking
  near 0.8 around step 73 before settling around 0.7").
- **Sample conversations:** "the same camping-supply scenario" → "camping-supply scenarios."

## I. Page-limit fit and consistency pass

- **Trimmed main body to the 4-page short-paper limit** (Limitations excluded). Condensed the
  experimental-setup paragraph, the "Quality over closure", "Aggressive anchoring", "Cross-task
  transfer", and "Selective deal-making" paragraphs, and the Discussion — no information removed;
  the Discussion now ends on page 4 and Limitations begins on page 5.
- **Standardized the replacement-model name** to `Qwen3.6-35B-A3B` across all result tables
  (previously `Qwen3.6-35B`), matching the appendix and eval configs.
- **Linked the ablation base to the baseline row:** the Transfer Ablation appendix now notes
  `Qwen3.6-35B-A3B` is the same model shown as a frontier baseline in Tables 1 and 5, served under
  the ablation's non-thinking renderer.
- **Removed em dashes** introduced in the new prose (Discussion closing, uniform-threshold
  appendix), replaced with parentheses/commas per standard usage.

---

## Known open items (not yet in `paper.tex`)

- **AHP results on the disjoint split** — split exists in code/data; the main results tables
  (Tables 1–2, 5–6) are not yet regenerated on it (see the `% TODO(rebuttal)` marker in
  Appendix A). Note this now requires retraining on the replacement base model, since the original
  checkpoints are no longer servable.
- **Anonymized repository URL** — availability footnote added; the actual anonymized repo link
  still needs to be created and substituted (see `% TODO(resubmission)` marker in §1).
- **Confidence intervals / significance tests** on the headline threshold-vs-surplus gaps
  (computable post-hoc from `logs/baseline_qwen` and `logs/baseline_gpt` episodes; not yet added).
- **Push `emnlp-overleaf` to Overleaf** — the Overleaf copy's content has been reconciled with this
  version (preamble/author block preserved); the change is in the working tree and still needs to be
  committed and pushed to the Overleaf remote by the author.
