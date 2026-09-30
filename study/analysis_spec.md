# Primary study analysis specification — draft before pilot

This records the analysis rules from `sc_math_reasoning_action_plan_v2.md`.
It is **not yet a frozen pre-registration**: checkpoint hashes, the final
evaluation seed, model-specific sampling settings, and the H2 decision rule
remain to be fixed before any model evaluation. No evaluation run is included
in this setup.

## Design and outcome

The primary set has 500 base problems, each in verbal/direct,
verbal/reversal, symbolic/direct, and symbolic/reversal forms. The unit of
pairing and bootstrap resampling is the base problem. A model response is
correct when its extracted final number exactly equals the stored integer
answer. Parse failures are incorrect. A response that hits the token limit
without a final answer is incorrect in the primary analysis; a sensitivity
analysis excludes truncated responses. Report parse failures and truncation
by configuration and condition.

For each configuration, calculate:

```
CATE_verbal   = accuracy(verbal, direct) - accuracy(verbal, reversal)
CATE_symbolic = accuracy(symbolic, direct) - accuracy(symbolic, reversal)
Delta_kw      = CATE_verbal - CATE_symbolic
```

Report all three with 2,000 paired bootstrap replicates over base problems.
Retain individual samples for mixed-effects models and average four sampled
responses per item for post-trained model accuracy. Base models use greedy
decoding. The primary comparison is `Delta_kw`; verbal CATE alone is the
comparison with Opedal et al.

## Hypotheses and tests

H1: `Delta_kw > 0` for each Qwen3 size. H2: `Delta_kw` does not materially
decrease with log parameter count from 0.6B through 14B; the 32B checkpoint
is reported separately because its training recipe differs. H3: thinking
mode increases `Delta_kw`. H4: the OLMo 3 Base → SFT → DPO → RLVR stage
contrasts increase `Delta_kw`, fitted separately for 7B and 32B. H5: first
errors concentrate at the comparison step and, conditional on a comparison
error, keyword-signature values are more frequent in verbal than symbolic
reversal items. H6: keyword effects increase with corpus log association
ratios, after the corpus keyword list is finalized.

Fit the plan's sample-level logistic mixed model for Qwen3:

```
correct ~ form * reversal * log(params) * mode + operation + n_steps + (1 | base_problem)
```

For OLMo 3, replace `log(params)` with `stage` and fit per size. Apply
Holm–Bonferroni to the six hypothesis families; use paired t-tests with
Benjamini–Hochberg correction as a secondary comparison. Report every
configuration, including null results.

H2 needs a pre-specified smallest meaningful decline and a confidence-
interval or equivalence decision rule. A nonsignificant negative slope alone
does **not** establish that the effect persists. Freeze that rule before the
pilot and full runs.

## Pilot and gates

Pilot 60 base problems × four renderings and 30 multi-comparison problems on
all configurations. Use the pilot to check floor and ceiling, token limits,
power for a five-point `Delta_kw`, and whether numbered steps change accuracy
by more than three points. If symbolic accuracy is below 50% for a
configuration, pilot the plan's keyword-neutral verbal control. If
consistent-verbal accuracy exceeds 97% for thinking models of at least 8B,
shift analysis weight to multi-comparison chains for those configurations.
Human QA follows the plan's 10-item gate then 90-item check separately for
each condition; any correction or label error requires a fix and a repeat.

Do not expose the final seed or generated evaluation items until all runs
finish. Commit model identifiers, revisions, prompt text, decoding settings,
generator revision, and the frozen specification before evaluation.
