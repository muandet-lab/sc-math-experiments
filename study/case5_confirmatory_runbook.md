# Running the locked Case 5 set

The [preregistration](case5_confirmatory_preregistration.md), [828-item benchmark](case5_confirmatory_items.jsonl), [72 prefill probes](case5_confirmatory_prefill.jsonl), [model manifest](case5_confirmatory_models.json), and [frozen scorer](score_case5_confirmatory.py) must be copied together to the VM. The prior pilots are not part of the confirmatory input. No smoke test is required or scheduled.

From `/mnt/scmath-data/src/sc-math-experiments`:

```sh
export HF_HOME=/mnt/scmath-data/cache/huggingface
export C5_ITEMS=study/case5_confirmatory_items.jsonl
export C5_PROBES=study/case5_confirmatory_prefill.jsonl
export C5_OUTPUT=/mnt/scmath-data/outputs/case5-confirmatory
mkdir -p "$C5_OUTPUT"
```

Run one model at a time. The runner appends after each batch and resumes only when input, checkpoint, sampling limit, and tensor parallelism match. The first six smaller checkpoints use one GPU:

```sh
for model in qwen_0_6b qwen_1_7b qwen_4b qwen_8b qwen_14b olmo_7b; do
  CUDA_VISIBLE_DEVICES=0 /mnt/scmath-data/venvs/vllm-cu130/bin/python \
    -m study.run_case5_confirmatory "$model" --items "$C5_ITEMS" \
    --output "$C5_OUTPUT/$model.raw.jsonl"
done
```

The two 32B BF16 models require multiple GPUs. This command assumes both visible A100s can fit the pinned model with 16,384 output tokens; that fit is **unverified** because the smoke test was skipped. If either fails for memory, preserve the error log and pause that model until a sufficiently large GPU configuration is available. Do not silently shorten the cap or quantize a checkpoint within this study.

```sh
for model in qwen_32b olmo_32b; do
  CUDA_VISIBLE_DEVICES=0,1 /mnt/scmath-data/venvs/vllm-cu130/bin/python \
    -m study.run_case5_confirmatory "$model" --items "$C5_ITEMS" \
    --tensor-parallel-size 2 --output "$C5_OUTPUT/$model.raw.jsonl"
done
```

Score only completed raw files. The scoring command refuses an incomplete grid and never overwrites output:

```sh
for model in qwen_0_6b qwen_1_7b qwen_4b qwen_8b qwen_14b qwen_32b olmo_7b olmo_32b; do
  /mnt/scmath-data/venvs/vllm-cu130/bin/python -m study.score_case5_confirmatory \
    --items "$C5_ITEMS" --raw "$C5_OUTPUT/$model.raw.jsonl" \
    --output "$C5_OUTPUT/$model.scored.jsonl"
done
```

Run the separate, fixed-prefix thinking-channel probability assay, again one checkpoint at a time:

```sh
for model in qwen_0_6b qwen_1_7b qwen_4b qwen_8b qwen_14b olmo_7b; do
  CUDA_VISIBLE_DEVICES=0 /mnt/scmath-data/venvs/vllm-cu130/bin/python \
    -m study.run_case5_prefill "$model" --probes "$C5_PROBES" \
    --output "$C5_OUTPUT/$model.prefill.jsonl"
done
for model in qwen_32b olmo_32b; do
  CUDA_VISIBLE_DEVICES=0,1 /mnt/scmath-data/venvs/vllm-cu130/bin/python \
    -m study.run_case5_prefill "$model" --probes "$C5_PROBES" \
    --tensor-parallel-size 2 --output "$C5_OUTPUT/$model.prefill.jsonl"
done
```

After copying all results back to the local checkout, install `matplotlib` in the analysis Python environment and R package `lme4` in the R environment, then generate six figures and regression data:

```sh
python3 -m study.analyze_case5_confirmatory \
  --items study/case5_confirmatory_items.jsonl \
  --results study/outputs/case5-confirmatory \
  --output study/outputs/case5-confirmatory/analysis
Rscript study/case5_confirmatory_regression.R \
  study/outputs/case5-confirmatory/analysis/regression.csv \
  study/outputs/case5-confirmatory/analysis/regression.txt
```

Prepare separate blind coding sheets for two coders, then compute agreement and adjudicate differences:

```sh
python3 -m study.code_case5_traces prepare \
  --results study/outputs/case5-confirmatory \
  --items study/case5_confirmatory_items.jsonl \
  --output study/outputs/case5-confirmatory/coding
python3 -m study.code_case5_traces agreement \
  --coder-a study/outputs/case5-confirmatory/coding/coder_a.csv \
  --coder-b study/outputs/case5-confirmatory/coding/coder_b.csv \
  --mapping study/outputs/case5-confirmatory/coding/mapping.jsonl \
  --output study/outputs/case5-confirmatory/coding/agreement.json
```

The coding sheets start empty. After both coders finish, make an `adjudicated.csv` with the same columns and resolved code for every trace; run `python3 -m study.code_case5_traces finalize --adjudicated ... --mapping ... --output ...` for final labels. Keep the mapping private from coders until their sheets are complete. The codebook is in the preregistration.
