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

The full-study multi-comparison and dose-response generators, checkpoint
manifest, bf16 generation runners, and human QA specified in the action plan
remain pending. The step-format accuracy pilot and corpus classifier validation
are also pending.
