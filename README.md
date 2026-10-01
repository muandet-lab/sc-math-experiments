# Spurious correlations in math reasoning

This is our experiment repository. It contains the [action plan](docs/action_plan_v2.md),
fresh matched-item generator, exploratory pilot analysis, tests, and VM notes.
The [Opedal et al. source repository](https://github.com/eth-lre/solving-biases)
is kept as a separate checkout at commit
`a7d858ff4129d3d5077bbbd0000c2796d3ce50bd`; its published instances
are not used as final evaluation items.

The current Compute Engine VM is `sc-math-2xa100` in `us-central1-f` under
project `rg-muandet-15801-1`. It has two A100 40 GB GPUs and a 1 TB data
disk mounted at `/mnt/scmath-data`. Keep model weights and generated outputs
on that disk. The VM checkout belongs at
`/mnt/scmath-data/src/sc-math-experiments`; the separate upstream checkout
already on the VM can remain at `/mnt/scmath-data/src/solving-biases`.

Run code checks and summarize the saved local exploratory pilot from the
repository root:

```sh
python3 -m unittest study.test_generate study.test_local_pilot study.test_shortcut_cases
python3 -m study.summarize_local_pilot
python3 -m study.summarize_gpu_pilot
```

`study/outputs/`, `study/private/`, virtual environments, and model weights
are local artifacts ignored by Git. The pilot reports are in
[`study/local_pilot_report.md`](study/local_pilot_report.md) and
[`study/gpu_pilot_report.md`](study/gpu_pilot_report.md). The exploratory
bf16 VM pilot runner is `study.run_gpu_pilot`. The full-study multi-GPU runner
and corpus-association measurement are still to be implemented; the Apple MLX
pilot runner is not the VM runner.

The five additional shortcut diagnostics (multiplicative reversal,
comparison versus transfer, sentence order, missing-premise solvability, and
changed-query template reuse) have matched-item generation and a thinking-only
GPU runner documented in [study/README.md](study/README.md).

## Sync committed code to the VM

The GitHub remote is
[`muandet-lab/sc-math-experiments`](https://github.com/muandet-lab/sc-math-experiments).
From the local checkout, commit changes and run `git push origin main`.
The raw pilot response JSONL files remain local; the generated confirmatory
benchmark and prefill JSONL files are tracked intentionally.

For an existing VM checkout, authenticate GitHub on the VM once, then add the
remote and fast-forward to the latest commit:

```sh
gh auth login -h github.com -p https -w
gh auth setup-git
git -C /mnt/scmath-data/src/sc-math-experiments remote add origin \
  https://github.com/muandet-lab/sc-math-experiments.git
git -C /mnt/scmath-data/src/sc-math-experiments fetch origin
git -C /mnt/scmath-data/src/sc-math-experiments merge --ff-only origin/main
```

If the VM checkout already has `origin`, skip `remote add`. For later updates,
run `git -C /mnt/scmath-data/src/sc-math-experiments pull --ff-only origin main`.
Commit or stash VM-local edits before pulling. Keep API credentials, model
weights, and raw response files out of Git.
