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
python3 -m unittest study.test_generate study.test_local_pilot
python3 -m study.summarize_local_pilot
```

`study/outputs/`, `study/private/`, virtual environments, and model weights
are local artifacts ignored by Git. The pilot report is in
[`study/local_pilot_report.md`](study/local_pilot_report.md). The full-study
bf16 multi-GPU runner and corpus-association measurement are still to be
implemented; the Apple MLX pilot runner is not the VM runner.

## Sync committed code to the VM

This repository is local-only for now. A Git bundle transports its committed
history without publishing a remote or copying model weights. After committing
new work on `main`, run **on the local machine**:

```sh
git bundle create /tmp/sc-math-experiments.bundle main
gcloud compute scp /tmp/sc-math-experiments.bundle sc-math-2xa100:~/sc-math-experiments.bundle \
  --project=rg-muandet-15801-1 --zone=us-central1-f
```

For the first transfer, run **on the VM**:

```sh
git clone ~/sc-math-experiments.bundle /mnt/scmath-data/src/sc-math-experiments
```

For later transfers, replace the bundle and run **on the VM**:

```sh
git -C /mnt/scmath-data/src/sc-math-experiments fetch ~/sc-math-experiments.bundle main
git -C /mnt/scmath-data/src/sc-math-experiments merge --ff-only FETCH_HEAD
```

The fast-forward merge leaves VM-local edits alone and fails if the two copies
diverge, so resolve that before continuing. Do not put API credentials or raw
private evaluation items into Git.
