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

Run the local checks with `python3 -m unittest study.test_generate study.test_local_pilot`.

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

To extend the same 20-item exploratory pilot to the five remaining Qwen3
sizes, use `study.run_qwen_scale_pilot`. It fixes `enable_thinking=True` and
the Qwen thinking sampling settings for every size. Each command writes a
separate JSONL file and refuses to overwrite an existing one. Run them from
the VM repository root after syncing the latest code:

```sh
export HF_HOME=/mnt/scmath-data/cache/huggingface
CUDA_VISIBLE_DEVICES=0 /mnt/scmath-data/venvs/vllm-cu130/bin/python -m study.run_qwen_scale_pilot 1.7b --output-dir /mnt/scmath-data/outputs
CUDA_VISIBLE_DEVICES=0 /mnt/scmath-data/venvs/vllm-cu130/bin/python -m study.run_qwen_scale_pilot 4b --output-dir /mnt/scmath-data/outputs
CUDA_VISIBLE_DEVICES=0 /mnt/scmath-data/venvs/vllm-cu130/bin/python -m study.run_qwen_scale_pilot 8b --output-dir /mnt/scmath-data/outputs
CUDA_VISIBLE_DEVICES=0 /mnt/scmath-data/venvs/vllm-cu130/bin/python -m study.run_qwen_scale_pilot 14b --output-dir /mnt/scmath-data/outputs
CUDA_VISIBLE_DEVICES=0,1 /mnt/scmath-data/venvs/vllm-cu130/bin/python -m study.run_qwen_scale_pilot 32b --output-dir /mnt/scmath-data/outputs
```

The 32B run uses tensor parallelism across the two 40 GB A100s and sets
vLLM's GPU memory utilization to 0.95; the other four use one GPU at 0.85.
This is a proposed pilot configuration, not a validated capacity claim: the
32B run still needs an actual startup and generation check on the VM.
All five runs use bf16, `max_model_len=4096`, 2,048 output tokens, and the
same selected five bases as the earlier pilot. The results therefore assess
this short pilot only; they do not implement the 500-base primary set or its
registered 16,384-token thinking limit. After copying the outputs locally,
pass their paths to `python3 -m study.summarize_gpu_pilot` to inspect the
per-cell accuracy and truncations.

The full-study multi-comparison and dose-response generators, checkpoint
manifest, bf16 generation runners, and human QA specified in the action plan
remain pending. The step-format accuracy pilot and corpus classifier validation
are also pending.
