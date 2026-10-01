# Keyword-consistency study setup

The separate upstream `solving-biases` checkout is pinned at
`a7d858ff4129d3d5077bbbd0000c2796d3ce50bd`.
Published upstream instances in `data/` are not used for evaluation.

Generate 60 pilot base problems (240 matched renderings):

```sh
python3 -m study.generate --count 60 --seed <private-integer> --output study/outputs/pilot.jsonl
```

For the planned primary set, use `--count 500` with a separate private seed.
The generator writes one JSON object per rendering, with all intermediate
values and the comparison step. It refuses to overwrite an existing output.
The four renderings of a base problem share every sentence except the
comparison sentence. Operations are balanced across the requested count;
counts divisible by four give exact balance. All quantities stay between
2 and 20. No model call or grammar correction is needed for these fixed
templates.

Run the local checks with
`python3 -m unittest study.test_generate study.test_local_pilot study.test_shortcut_cases study.test_case5_diagnostics`.

An exploratory local pilot uses five base problems of at most three steps,
each in verbal direct, verbal reversal, symbolic direct, and symbolic reversal
form. See [local_pilot_report.md](local_pilot_report.md) for the completed
results. The saved raw responses can be summarized with:

```sh
python3 -m study.summarize_local_pilot
```

To rerun model generation on an Apple Silicon machine, install
`mlx-lm==0.31.3` in a local environment and use `study.run_local_pilot`
with the pinned 4-bit conversions in `study/private/models/`. That runner
does not run on the Linux A100 VM.

The runner resumes existing files; outputs and model weights are ignored by
Git. The pilot is exploratory and its seed and outputs are excluded from the
planned final evaluation.

The bf16 A100 pilot uses the same five base problems and writes every raw
response, including any thinking text, to JSONL. With the vLLM 0.30.0 CUDA 13
environment on the VM, run one configuration at a time from the repository root:

```sh
export HF_HOME=/mnt/scmath-data/cache/huggingface
CUDA_VISIBLE_DEVICES=0 /mnt/scmath-data/venvs/vllm-cu130/bin/python -m study.run_gpu_pilot \
  qwen --mode non-thinking \
  --output /mnt/scmath-data/outputs/qwen3-0.6b-bf16-nonthinking-pilot.jsonl
CUDA_VISIBLE_DEVICES=0 /mnt/scmath-data/venvs/vllm-cu130/bin/python -m study.run_gpu_pilot \
  olmo --mode native-thinking \
  --output /mnt/scmath-data/outputs/olmo3-7b-think-bf16-pilot.jsonl
```

The runner resolves and records each official Hugging Face model commit before
loading weights. It refuses to overwrite outputs. This is a functional pilot,
not the pre-registered final study; its 20 matched renderings per model are
insufficient for the planned hypothesis tests.

The completed bf16 results and error traces are discussed in
[gpu_pilot_report.md](gpu_pilot_report.md). After copying the three JSONL
outputs to `study/outputs/` locally, recompute its table with:

```sh
python3 -m study.summarize_gpu_pilot
```

The full-study multi-comparison and dose-response generators, checkpoint
manifest, bf16 generation runners, and human QA specified in the action plan
remain pending. The step-format accuracy pilot and corpus classifier validation
are also pending.

## Additional shortcut cases (2–6)

`study.generate_shortcut_cases` creates fresh matched pairs for the five
diagnostics beyond additive keyword reversal. Its `--count` is the number of
base pairs, so each output has twice that many prompts. These are controlled
diagnostics, not the action plan's frozen primary evaluation set. See
[shortcut_cases_spec.md](shortcut_cases_spec.md) for the manipulation and
measurement rules and their limits.
The completed thinking-mode pilot and its corrected scoring are discussed in
[shortcut_cases_pilot_report.md](shortcut_cases_pilot_report.md).

| Case | Control → challenge | Gold answer |
| --- | --- | --- |
| 2. Multiplicative reversal | Direct vs. reversed “times as many” or reciprocal relation | Same integer |
| 3. Comparison vs. transfer | Target receives/gives away an amount vs. target has more/fewer than a source | Same integer |
| 4. Sentence order | Two static comparisons in dependency order vs. the same facts reversed | Same integer |
| 5. Solvability prior | Complete problem vs. initial numerical premise removed | Integer vs. “cannot be determined” |
| 6. Solution template | Same facts, query target vs. query source | Different integers |

For an exploratory smoke set on the VM, generate ten pairs per case with a
seed that will **not** be reused for final evaluation. First sync the latest
code to the VM, then run from its repository root:

```sh
for case in 2 3 4 5 6; do
  /mnt/scmath-data/venvs/vllm-cu130/bin/python -m study.generate_shortcut_cases \
    "$case" --count 10 --seed 20261001 \
    --output "/mnt/scmath-data/outputs/shortcut-case${case}-items.jsonl" || break
done
```

Run each input with either `qwen` (Qwen3-0.6B, `enable_thinking=True`) or `olmo`
(OLMo 3 7B Think in its native thinking mode). The CLI has no non-thinking
option. From the VM repository root, these commands run both models on all
five cases, one process at a time:

```sh
export HF_HOME=/mnt/scmath-data/cache/huggingface
for model in qwen olmo; do
  for case in 2 3 4 5 6; do
    CUDA_VISIBLE_DEVICES=0 /mnt/scmath-data/venvs/vllm-cu130/bin/python \
      -m study.run_shortcut_cases "$model" \
      --input "/mnt/scmath-data/outputs/shortcut-case${case}-items.jsonl" \
      --output "/mnt/scmath-data/outputs/${model}-shortcut-case${case}-thinking.jsonl" || break 2
  done
done
```

The runner resolves and records the model revision, refuses to overwrite
existing output, and saves every raw response. Default limits
are 2,048 generated tokens and a 4,096-token context; use `--max-tokens` and
`--max-model-len` together to pilot longer traces. Copy results locally and
summarize one or more response files with:

```sh
python3 -m study.summarize_shortcut_cases study/outputs/*shortcut-case*-thinking.jsonl
```

The summary reports paired accuracy and, for case 5, numerical guessing on
unsolvable questions; for case 6, whether a changed-query answer repeats the
original answer. Case 3 necessarily changes how the known starting value is
attached to the queried agent, so its gap is a comparison-versus-transfer
diagnostic rather than a pure keyword effect. Human QA, larger pilot sizes,
and checks for floor, ceiling, parsing, and truncation are required before
inferential use.

For the next **case-5-only pilot**, use new output filenames because the answer
prompt has changed since the first pilot. A longer generation limit may reduce
unfinished thinking traces; test GPU fit and runtime with one model at a time:

```sh
for model in qwen olmo; do
  CUDA_VISIBLE_DEVICES=0 /mnt/scmath-data/venvs/vllm-cu130/bin/python \
    -m study.run_shortcut_cases "$model" \
    --input /mnt/scmath-data/outputs/shortcut-case5-items.jsonl \
    --max-tokens 4096 --max-model-len 8192 \
    --output "/mnt/scmath-data/outputs/${model}-shortcut-case5-format-v2.jsonl" || break
done
```

The output records `parse_status` alongside the raw response and finish
reason. After copying the files locally, run the summarizer on the new paths.
It reports `unparsed_completed` separately from `truncations`; inspect those
rows before interpreting the accuracy gap. The first and second pilots have
different prompts and token limits and should be reported separately.

## Case-5 behavioral diagnostics

[case5_diagnostics_spec.md](case5_diagnostics_spec.md) describes three tests of
missing-number location, implicit versus explicit unknowns, and ordinary
solving versus answerability assessment. The nine variants of each base problem
are independent single-turn conversations.
The Qwen and OLMo results and scorer corrections are recorded in
[case5_diagnostics_pilot_report.md](case5_diagnostics_pilot_report.md).

For an exploratory 10-base pilot on the VM, use a seed that will not be
reused for later evaluation:

```sh
/mnt/scmath-data/venvs/vllm-cu130/bin/python -m study.generate_case5_diagnostics \
  --count 10 --seed 20261003 \
  --output /mnt/scmath-data/outputs/case5-diagnostics-pilot-items.jsonl
for model in qwen olmo; do
  CUDA_VISIBLE_DEVICES=0 /mnt/scmath-data/venvs/vllm-cu130/bin/python \
    -m study.run_case5_diagnostics "$model" \
    --input /mnt/scmath-data/outputs/case5-diagnostics-pilot-items.jsonl \
    --output "/mnt/scmath-data/outputs/${model}-case5-diagnostics-pilot.jsonl" || break
done
```

The diagnostic runner defaults to `--max-tokens 4096 --max-model-len 8192`,
uses one visible A100 40 GB GPU with `gpu_memory_utilization=0.85`, and records
the limits in each output row. These are the saved pilot settings, not a known
GPU maximum. If many responses still truncate, test `--max-tokens 8192
--max-model-len 16384` on a small set first and use new output filenames.
After copying response files locally, summarize them with:

```sh
python3 -m study.summarize_case5_diagnostics \
  study/outputs/*-case5-diagnostics-pilot.jsonl
```

Inspect raw completed errors before drawing a mechanism conclusion. For a
generalization run, generate new base problems with a fresh private seed and
freeze the prompt, limits, model revisions, and scoring rules before inference.

### Assumption-instruction follow-up

The final exploratory pilot adds one instruction against unstated assumptions
to only three existing variants: omitted initial amount, complete problem, and
net change with an unknown initial amount. It reuses the original item file,
so there is no new generation step. Run one model at a time on the VM:

```sh
for model in qwen olmo; do
  CUDA_VISIBLE_DEVICES=0 /mnt/scmath-data/venvs/vllm-cu130/bin/python \
    -m study.run_case5_assumption_pilot "$model" \
    --input /mnt/scmath-data/outputs/case5-diagnostics-pilot-items.jsonl \
    --output "/mnt/scmath-data/outputs/${model}-case5-assumption-pilot.jsonl" || break
done
```

Each file contains 30 fresh single-turn responses and records the complete
system prompt. The runner pins each model to the revision in its saved baseline:
Qwen `c1899de289a04d12100db370d81485cdf75e47ca` and OLMo
`d97e442d7cc678210054dbcc9b440894d62c89a4`. The defaults remain 4,096
generated tokens and an 8,192-token context. The runner refuses to overwrite
an existing output. After copying
the new files locally, compare each model with its original 90-row baseline:

```sh
python3 -m study.summarize_case5_assumption_pilot \
  --baseline study/outputs/qwen-case5-diagnostics-pilot.jsonl \
  --intervention study/outputs/qwen-case5-assumption-pilot.jsonl
python3 -m study.summarize_case5_assumption_pilot \
  --baseline study/olmo-case5-diagnostics-pilot.jsonl \
  --intervention study/outputs/olmo-case5-assumption-pilot.jsonl
```

See [case5_diagnostics_spec.md](case5_diagnostics_spec.md) for the intervention
wording and interpretation limits.
