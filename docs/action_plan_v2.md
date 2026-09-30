# Action Plan v2: Spurious Correlations in Math Reasoning Across Model Scale

**Evaluation-only study (no training, no fine-tuning)**

**Changes from v1:**
- The **primary task is now the keyword-consistency effect** in arithmetic word problems (Opedal et al., ICML 2024), using a generator that produces fresh, paired items with ground truth for every intermediate step.
- **BrokenMath moves to a secondary study** of sycophancy.
- A **symbolic control** is added to separate the keyword shortcut from the difficulty of reversing a relation.
- The **training association is now measured directly** in OLMo 3's open training data.

---

## 0. Research question

Do documented spurious correlations (SCs) stop affecting reasoning as models get larger and more capable, or do they persist, possibly in more complex forms?

This study targets one SC with a documented causal behavioral effect in LLMs: the association between a **relational keyword** and an **arithmetic operation** ("more" → add, "fewer" → subtract). It measures that effect across:

- **model size**: the Qwen3 dense line
- **reasoning mode**: thinking vs. non-thinking
- **training stage**: OLMo 3 Base → SFT → DPO → RLVR

It separates the keyword-specific effect from the general difficulty of reversing a relation, locates errors at the level of individual reasoning steps, and tests whether the strength of the effect tracks how often each phrasing appears in the training data.

---

## 1. Prior evidence this plan builds on

From Opedal et al. (ICML 2024), Table 2, consistency-bias columns:

| Finding | Evidence |
|---|---|
| The effect is robust | 20 of 23 CATEs significant (§5.2) |
| It is not removed by capability | CoT: GPT-3.5 Turbo 1.4 (n.s.) but GPT-4 Turbo **18.0** (90.4 vs. 72.4); Llama-2 base 7B/13B/70B: 10.4 / 21.6 / 16.2 |
| Chain-of-thought amplifies it | e.g. Llama-2-70B-Chat: 1.6 (direct) → 21.4 (CoT) |
| Instruction tuning may amplify it | Stated for transfer vs. comparison bias in §5.3; the authors say it "seems to be the case" for consistency too (remark in the text, no separate test) |
| Training association: **weak evidence** | 15 consistent vs. 3 inconsistent comparison problems (n = 18) in MAWPS / ASDiv-A / SVAMP (§6, fn. 10); the authors could not verify the models' training data |

**Gaps this study fills:**
1. No results for modern reasoning models or thinking modes.
2. No systematic scaling within a single model family.
3. The keyword shortcut is not separated from the reversal difficulty.
4. The training association has not been measured in a model's actual training data.

---

## 2. Hypotheses (pre-register before any run)

| ID | Hypothesis | Competing prediction |
|---|---|---|
| **H1** | The keyword-specific effect (Δ_kw, §7.2) is > 0 for every Qwen3 size. | The effect is fully explained by the reversal difficulty (Δ_kw = 0). |
| **H2** | Δ_kw does **not** shrink with log(parameters) across Qwen3 0.6B–14B. | Δ_kw shrinks with scale ("the SC disappears"). |
| **H3** | Thinking mode **increases** the consistency effect relative to non-thinking mode, extending Opedal et al.'s CoT result. | Long-form reasoning removes the effect. |
| **H4** | In OLMo 3, the effect grows from Base to post-trained stages (SFT/DPO). | The effect is set in pretraining and flat across stages. |
| **H5** | Errors concentrate at the **comparison step**, and erroneous values match the keyword-predicted operation more often in the verbal condition than in the symbolic condition. | Errors are spread across steps independently of the manipulation. |
| **H6** | Across relational keywords, the effect size increases with how strongly each keyword is associated with its "default" operation in OLMo 3's training data. | No relationship between measured training association and effect size. |

Correct for multiple comparisons across H1–H6 with Holm–Bonferroni. H1 and H2 are the primary hypotheses.

---

## 3. Target spurious correlation

### 3.1 Definition (Ye et al., TMLR 2026, Def. 2.1 / Fig. 2 / §2.3)

- **Label *y*:** the arithmetic operation the problem requires.
- **Spurious attribute *a*:** the relational keyword and the operation it suggests ("more" → +, "fewer" → −, "times as many" → ×).
- **Bias-aligned (consistent):** the keyword suggests the required operation. "Eli has 5 fewer tokens than Nora" → Eli = Nora − 5.
- **Bias-conflicting (inconsistent):** the keyword suggests the opposite operation. "Nora has 5 more tokens than Eli" → Eli = Nora − 5.
- **Training association:** consistent phrasings are hypothesized to be more frequent in training data. The existing evidence is weak (n = 18). This study measures it directly in OLMo 3 (§6).

### 3.2 The confound the design must separate

Inconsistent statements also require **reversing the relation**: the unknown quantity is the referent rather than the subject. Lewis and Mayer (1987) attribute the human effect to this extra step. A model that maps "more" to + and a model that fails to reverse the relation make the **same wrong computation** (Nora + 5 = 17). So the rate of that wrong answer cannot tell the two apart.

**Solution:** a symbolic control condition that keeps the direction of the relation and removes the keyword (§5.2). The keyword-specific effect is then a difference-in-differences.

### 3.3 Co-occurring factors held constant

| Factor | Control |
|---|---|
| Transfer vs. comparison bias (Opedal §5.3) | Every item contains the same number of comparison predicates within a pair; transfer and rate predicates are identical within pairs |
| Carry effect | Same numbers within a pair; small number range (2–20) as in Opedal |
| Irrelevant information | None in the primary task |
| Sycophancy | No user assertions in the primary task (BrokenMath, §9, handles this separately) |
| Agent names, entities, templates | Identical within a pair; only the relational sentence varies |

---

## 4. Models

### 4.1 Qwen3 — size axis

| Checkpoint | Post-training recipe | Evaluated as |
|---|---|---|
| Qwen3-0.6B, 1.7B, 4B, 8B, 14B | Strong-to-weak **distillation** | thinking, non-thinking |
| Qwen3-32B | Four-stage pipeline with **reasoning RL** | thinking, non-thinking |
| Qwen3-Base (every size released) | Pretraining only | direct + zero-shot CoT (Opedal prompts) |

- Use the **original hybrid release** (April 2025) and pin the Hugging Face commit hashes.
- **Scaling fits use 0.6B–14B only**, which share one recipe. Report 32B separately as the RL comparison point.
- Exclude the MoE variants.
- Check which Qwen3-Base sizes were actually released before committing.

### 4.2 OLMo 3 — training-stage axis, auditable data

| Size | Stages |
|---|---|
| Olmo 3 7B | Base, Think-SFT, Think-DPO, Think (RLVR) |
| Olmo 3 32B | Base, Think-SFT, Think-DPO, Think (RLVR) |
| Olmo 3.1 32B Think | final (extended RL) |

Unlike in v1, **Base checkpoints are included**, because Opedal-style arithmetic word problems work with base models using the paper's prompt formats.

### 4.3 Configuration count (primary task)

- Qwen3 post-trained: 6 sizes × 2 modes = 12
- Qwen3-Base: up to 6 × 2 prompt modes = up to 12
- OLMo 3: 9 checkpoints; post-trained ones in their native mode, Base in direct + CoT

---

## 5. Primary task: keyword consistency

### 5.1 Generator

**Starting point:** `github.com/eth-lre/solving-biases` (`generate_consistency_model.py`, `generate_data.py`). Pin the commit.

**Changes:**

1. **Fresh seeds.** Never evaluate on the instances in the repo's `data/` folder, which have been public since 2024. Keep the evaluation seed private until all runs are finished.
2. **Replace the GPT-3.5 Turbo correction step.** Use a pinned open instruction model from **outside** the evaluated families to correct grammar. Keep Opedal's integrity checks (sentence count, relational terms preserved, numbers unchanged) and add a check that the **consistency label is unchanged**. Opedal et al. found the correcting model sometimes rewrote inconsistent statements as consistent ones.
3. **Add the symbolic rendering** of the comparison sentence (§5.2).
4. **Add multi-comparison chains** for difficulty (§5.3).
5. **Store for every item:** the full mental model (sequence of logical forms), every intermediate value, which step is the comparison, the consistency label, and the keyword used.

**Quality assurance:** follow Opedal et al.'s App. A.2 protocol. Three annotators check 10 control items; if all pass, 90 more are checked (30 each). The target is a 0% error rate on both the correction and consistency criteria. Repeat for every new condition (symbolic, multi-comparison).

### 5.2 Core design: 2 × 2 within-item

Every **base problem** is rendered in four versions that differ only in the comparison sentence:

| | Unknown is the subject (no reversal) | Unknown is the referent (reversal needed) |
|---|---|---|
| **Verbal** | Consistent: "Eli has 5 fewer tokens than Nora." | Inconsistent: "Nora has 5 more tokens than Eli." |
| **Symbolic** | "Eli's tokens = Nora's tokens − 5." | "Nora's tokens = Eli's tokens + 5." |

- Cover **all four operations** (+, −, ×, ÷) via additive and multiplicative comparisons, balanced, as in Opedal §5.2.
- Chains of **1–5 reasoning steps** with exactly one comparison, placed at a random position within the chain.
- **500 base problems × 4 versions = 2,000 items**, matching Opedal's 500 pairs per test.

### 5.3 Difficulty extension: multi-comparison chains

Single-comparison items are likely near ceiling for Qwen3 in thinking mode.

- **300 base problems** with **2–4 comparison predicates** in chains of 4–8 steps. Each comparison is independently rendered consistent or inconsistent, verbal only.
- This gives a within-problem, step-level test: are errors more likely at inconsistent comparison steps than at consistent ones in the same problem?

### 5.4 Dose-response set (for H6)

- **200 base problems** in the verbal inconsistent form, rendered with a **range of relational keywords** that plausibly vary in how strongly they are associated with an operation.
  - Additive: "more / fewer," "extra," "additional," "less," "greater than," "short of."
  - Multiplicative: "times as many," "a fraction of."
- The keyword list is **finalized after the corpus measurement** (§6.2), choosing keywords that span a wide range of measured association strength.

### 5.5 Prompts and answer extraction

| Model type | Prompt |
|---|---|
| Base, direct | Opedal format: `Q: {problem}\nA: The answer (Arabic numerals) is ` |
| Base, CoT | Opedal two-stage zero-shot CoT ("Let's think step by step", then re-prompt for the answer) |
| Post-trained, both modes | Chat template + an instruction to show the work as numbered steps and end with `Final answer: <number>` |

- **Step format for post-trained models:** request `Step i: <quantity> = <value>` in the final answer (after the thinking block, in thinking mode). Pilot whether this requirement changes accuracy by more than 3 points; if it does, extract steps with a validated LLM parser instead.
- **Decoding:** base models use greedy decoding for comparability with Opedal Table 2. Post-trained models use each model card's recommended sampling settings, with 4 samples per item. Record all settings.
- **Token limits:** direct 32; base CoT 512; non-thinking 2,048; thinking 16,384. Truncation handling is pre-registered in §8.3.

---

## 6. Measuring the training association (OLMo 3)

This is the step that moves the study from "effect consistent with an SC" to "SC with a measured training association."

### 6.1 Corpora

- Olmo 3 pretraining data (Dolma 3), focusing on the math and educational subsets.
- The Olmo 3 **Think** SFT, DPO and RLVR datasets, each counted **separately**, so the association can be compared with the stage-wise effect (H4).

### 6.2 Procedure

1. **Retrieve** candidate comparison sentences matching relational patterns: "`<agent>` has `<N>` more/fewer/… `<entity>` than `<agent>`", "… times as many …", and synonyms. Use Ai2's corpus-search tooling if it is available for Dolma 3; otherwise sample shards uniformly.
2. **Classify** each retrieved problem as consistent or inconsistent. This needs the question, because consistency depends on which quantity is unknown. Use an LLM classifier validated on 300 hand-labelled examples, with a target agreement of at least 90%.
3. **Output:** the consistent : inconsistent ratio **per keyword and per corpus**, with bootstrap confidence intervals.

### 6.3 Uses

- **H6:** regress each keyword's effect size on its measured log-ratio, across keywords.
- **H4:** compare how the ratio changes from pretraining to SFT, DPO and RLVR data against how the effect changes across those checkpoints.
- A direct test of Opedal et al.'s untested hypothesis, with a sample in the thousands instead of 18.

---

## 7. Metrics and analysis (pre-registered)

### 7.1 Item-level outcome

- **Correct** = 1 if the extracted final answer equals the ground truth.
- With 4 samples per item for post-trained models, the item score is the mean. Keep sample-level data for the mixed-effects models.

### 7.2 Primary effects

For each configuration, over the matched 2 × 2 items:

- **Verbal consistency effect** (Opedal's CATE): CATE_verbal = Acc(verbal, consistent) − Acc(verbal, inconsistent)
- **Reversal effect:** CATE_symbolic = Acc(symbolic, no reversal) − Acc(symbolic, reversal)
- **Keyword-specific effect (primary SC estimate):**

> **Δ_kw = CATE_verbal − CATE_symbolic**

- Report all three. CATE_verbal alone is comparable with Opedal Table 2: match non-thinking and direct to the Direct half, and thinking and CoT to the CoT half.

### 7.3 Scaling and stage tests

**Model S1 (Qwen3 0.6B–14B):** mixed-effects logistic regression

```
correct ~ form * reversal * log(params) * mode + operation + n_steps + (1 | base_problem)
```

- **H1:** the `form × reversal` interaction (the keyword-specific effect) is > 0.
- **H2:** the `form × reversal × log(params)` interaction. H2 predicts ≥ 0 or not significant; the competing prediction is negative (shrinking).
- **H3:** the `form × reversal × mode` interaction.

**Model S2 (OLMo 3):** same structure, with `stage` (Base / SFT / DPO / RLVR) in place of `log(params)`, fitted per size. **H4:** stage contrasts on `form × reversal`.

**Uncertainty:** paired bootstrap over base problems (2,000 replicates) for CATEs and Δ_kw. Paired t-tests with Benjamini–Hochberg correction as a secondary analysis, for direct comparability with Opedal et al.

### 7.4 Step-level metrics (H5)

Using the generator's ground-truth intermediate values:

- **First-error step:** the earliest step whose value differs from ground truth.
- **P(first error = comparison step)**, by condition, compared with the base rate expected from chain length.
- **Keyword-signature error rate:** P(value at the comparison step equals the keyword-predicted value | error at the comparison step), e.g. 17 instead of 7. Compare **verbal vs. symbolic** at matched reversal. Taken alone this rate cannot separate the two explanations (§3.2); its verbal-minus-symbolic difference can.
- **Error propagation:** given a wrong value at the comparison step, the fraction of later steps that are computed correctly *from the wrong value*. This distinguishes a coherent wrong premise from scattered errors.
- **Multi-comparison chains (§5.3):** within-problem logistic regression of step error on step consistency, with step position as a covariate.

### 7.5 Trace-level metric (thinking mode, secondary)

- **Rewriting detection:** does the thinking trace restate an inconsistent relation in the consistent form before computing? Label this with a judge validated on 50 traces. Explicit rewriting followed by a correct answer is the reversal strategy working; inconsistent items answered wrongly with no rewriting point toward the keyword shortcut.

### 7.6 Reporting standards

Report every configuration including nulls; report truncation and parse-failure rates per cell; release prompts, generator commit, seeds (after the runs), raw outputs, and analysis code.

---

## 8. Procedures

### 8.1 Contamination and exposure checks

| Check | Method | Expected result |
|---|---|---|
| Exposure to the released Opedal data | Per model: perplexity on the repo's `data/` instances vs. fresh instances from the same templates | No difference. If published instances show clearly lower perplexity, report it (fresh items remain valid) |
| Calibration of the completion test | Partial-prompt completion (40/60/80% prefixes) on fresh items | Near-zero completion, confirming the test's baseline |
| Paired design | Consistent and inconsistent items share everything except one sentence | General familiarity with the template cancels in the CATE |

### 8.2 Pilot

- 60 base problems × 4 versions, plus 30 multi-comparison problems, on all configurations.
- Estimates for power analysis (sample sizes are fixed from the pilot variance, targeting 80% power for a 5-point Δ_kw at the smallest size); ceiling and floor checks per configuration; token-limit adequacy; parser validation.

### 8.3 Truncation rule

A response that hits the token limit without a final answer is coded **incorrect** in the primary analysis. Sensitivity analysis: exclude truncated responses and report whether conclusions change. Report truncation rates per configuration × condition. If truncation differs between consistent and inconsistent versions, that is itself a finding and is reported.

### 8.4 Serving

Use vLLM with bf16 weights on a single GPU where possible. Pilot 32B on one
96GB G4 GPU (`g4-standard-48`) and one 80GB A100 (`a2-ultragpu-1g`), in that
order of memory headroom, because reliable simultaneous allocation of two
A100s is difficult. Start with concurrency 1 and keep the four samples per
item sequential if necessary. Verify that the registered 16,384-token thinking
limit, not merely model loading, fits; record peak memory, throughput,
truncation, and out-of-memory failures. If neither single-GPU option supports
bf16 at the registered limit, report the infrastructure limit and treat any
quantized, CPU-offloaded, or shorter-context run as a separate configuration.
Do not pool its results with the primary bf16 comparison.

---

## 9. Secondary study: sycophancy on BrokenMath

Runs after the primary task. It is condensed from v1, and its design is unchanged.

- **Items:** BrokenMath public `benchmark` split. The 451 proof problems as released (false) plus their originals (true) give 902 items.
- **Framings:** F0 "Prove X" and F1 "I'm confident X is true. Prove X." The well-posedness prior cancels in F1 − F0.
- **Metrics:** signal detection. d′ is capability; c is acceptance bias. The primary effect is **Δc = c(F1) − c(F0)**, plus a capability-conditioned analysis (unsolved stratum) and cue acknowledgment in thinking traces.
- **Judge:** the three-vote GPT-5-mini protocol, validated on about 100 human-labelled responses, oversampling small models. A second rubric is needed for true statements.
- **Models:** Qwen3 ≥ 8B and OLMo 3 post-trained checkpoints. Smaller models are expected at floor and are not run unless the primary-task results motivate it.
- **Cost:** about 400k judge calls at full scale. Pilot on 100 problems first.

---

## 10. Compute budget (estimates, revise after the pilot)

| Component | Items | Generations | Judge calls |
|---|---|---|---|
| Primary 2×2 | 2,000 | ~12 post-trained configs × 2,000 × 4 samples ≈ 96k; Base/OLMo-Base greedy ≈ 40k | none |
| Multi-comparison | 1,200 (300 × 4 renderings) | ~60k | none |
| Dose-response | ~1,600 (200 × ~8 keywords) | ~70k | none |
| Contamination checks | — | ~30k (short) | none |
| Corpus classifier | — | ~20k classifications | none |
| BrokenMath (secondary) | 902 × 2 framings | ~60k (≥ 8B only) | ~180k–400k |

The primary task needs **no judge**, since answers are exact numbers. Its cost is dominated by thinking-mode runs at 14B and 32B.

---

## 11. Timeline (indicative)

| Week | Work | Deliverable |
|---|---|---|
| 1 | Pre-registration; pin checkpoints; set up the generator; replace the correction step | Pre-registration doc; generator fork |
| 2 | Symbolic and multi-comparison extensions; QA (§5.1); pilot | QA report; pilot tables; power analysis |
| 2–3 | Corpus retrieval and classifier validation (§6) | Association estimates per keyword and corpus |
| 3 | Finalize the dose-response keyword list | Frozen item set |
| 3–5 | Full primary runs + contamination checks | Raw outputs |
| 6 | Analysis (§7) | Figures and tables |
| 6–8 | BrokenMath secondary study (pilot → full) | Secondary results |
| 8–9 | Write-up | Draft paper |

---

## 12. Decision points

| After | Condition | Action |
|---|---|---|
| Generator QA | Any consistency-label or correction error in 100 items | Fix the templates or checks, then re-run QA |
| Pilot | Symbolic condition accuracy < 50% for a configuration | The symbolic form itself is too hard; add a verbal keyword-neutral control ("differ by 5; Nora has the larger count") and pilot again |
| Pilot | Consistent-verbal accuracy > 97% for thinking ≥ 8B | Shift the weight of the analysis to multi-comparison chains for those configurations |
| Pilot | Required step format shifts accuracy by > 3 points | Use a validated LLM parser on free-form output instead |
| Corpus analysis | Classifier agreement < 90% | Add annotation examples; restrict to high-precision patterns |
| Corpus analysis | Measured association range across keywords is too narrow to test H6 | Report H6 as untestable; keep the stage-wise ratio comparison |
| Exposure check | Published Opedal instances clearly memorized | Report it; keep the fresh items |

---

## 13. Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Symbolic form has its own difficulty (unfamiliar notation) | Δ_kw biased | The difference-in-differences cancels the main effect of form; the pilot checks for floor; fallback keyword-neutral verbal control |
| Ceiling for large thinking models | No measurable effect | Multi-comparison chains; report the ceiling as a result |
| Recipe break at Qwen3-32B | Size confounded with training method | Fit 0.6B–14B; report 32B separately |
| OLMo has only 2 sizes | No OLMo scaling claim | OLMo is used for the stage axis and corpus measurement only |
| The correcting LLM alters the manipulation | Mislabelled items | Label-preservation checks; QA protocol; a correcting model from outside the evaluated families |
| Corpus retrieval misses paraphrases | Association underestimated | Report retrieval patterns; validate recall on a hand-labelled sample |
| Imposing a step format alters behavior | Step-level metrics biased | Pilot check; parser fallback |
| Sampling variance in thinking modes | Noisy per-configuration estimates | 4 samples per item; paired bootstrap |
| Single-GPU memory for 32B at the full thinking limit | A100 80GB may load weights but run out of KV-cache memory | Pilot G4 96GB and A100 80GB at concurrency 1; report any separate precision or context fallback explicitly |

---

## 14. Deliverables

1. Pre-registration (hypotheses, metrics, exclusion and truncation rules)
2. Extended generator (symbolic, multi-comparison, dose-response), open source
3. Released evaluation instances (after the runs)
4. Measurements of the training association per keyword and per OLMo 3 corpus/stage
5. Results:
   - CATE_verbal, CATE_symbolic and Δ_kw vs. log(params) for Qwen3, by mode
   - Δ_kw across Base → SFT → DPO → RLVR for OLMo 3
   - Step-level error localization and keyword-signature errors
   - Effect size vs. measured training association across keywords
   - Comparison with Opedal et al. Table 2, matched by mode
6. Secondary: sycophancy decomposition on BrokenMath
7. Write-up linking the results to the Song et al. taxonomy and Ye et al.'s definition

---

## 15. Out of scope for v2

- A GSM-IC / NoOp distraction cue family (contested once ambiguous items are removed)
- The transfer vs. comparison bias (Opedal §5.3) as a second target
- The well-posedness prior as a primary target
- In-context injection of an arbitrary shortcut (Tang et al., 2023)
- Perceptual SCs with Qwen3-VL

---

## References

**Surveys and definitions**
- Song, Han & Goodman. *Large Language Model Reasoning Failures.* TMLR 2026. arXiv:2602.06176
- Ye et al. *The Clever Hans Mirage: A Comprehensive Survey on Spurious Correlations in Machine Learning.* TMLR 2026

**Primary task**
- Opedal, Stolfo, Shirakami, Jiao, Cotterell, Schölkopf, Saparov & Sachan. *Do Language Models Exhibit the Same Cognitive Biases in Problem Solving as Human Learners?* ICML 2024. arXiv:2401.18070. Code: github.com/eth-lre/solving-biases
- Lewis & Mayer. *Students' Miscomprehension of Relational Statements in Arithmetic Word Problems.* Journal of Educational Psychology, 1987
- Hegarty, Mayer & Monk. *Comprehension of Arithmetic Word Problems: A Comparison of Successful and Unsuccessful Problem Solvers.* Journal of Educational Psychology, 1995
- Opedal, Stoehr, Saparov & Sachan. *World Models for Math Story Problems (MathWorld).* Findings of ACL 2023

**Secondary task**
- Petrov, Dekoninck & Vechev. *BrokenMath: A Benchmark for Sycophancy in Theorem Proving with LLMs.* arXiv:2510.04721
- Sharma et al. *Towards Understanding Sycophancy in Language Models.* ICLR 2024

**Models**
- Qwen Team. *Qwen3 Technical Report.* arXiv:2505.09388
- Ai2. *Olmo 3* model cards and release notes (Nov/Dec 2025)

**Contamination**
- *Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination.* arXiv:2507.10532
- *Spurious Rewards Paradox.* arXiv:2601.11061

**Related**
- Patel et al. *SVAMP.* NAACL 2021
- Shi et al. *GSM-IC.* ICML 2023
- Mirzadeh et al. *GSM-Symbolic.* ICLR 2025
- Razeghi et al. *Impact of Pretraining Term Frequencies on Few-Shot Numerical Reasoning.* Findings of EMNLP 2022
- Tang et al. *Large Language Models Can Be Lazy Learners.* Findings of ACL 2023
