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

The intended public GitHub remote is
[`muandet-lab/sc-math-experiments`](https://github.com/muandet-lab/sc-math-experiments),
pending GitHub authentication and creation. Until then, transfer committed
code as a Git bundle. On the local machine, after committing new work on
`main`, run:

```sh
git bundle create /tmp/sc-math-experiments.bundle main
gcloud compute scp /tmp/sc-math-experiments.bundle \
  sc-math-2xa100:~/sc-math-experiments.bundle \
  --project=rg-muandet-15801-1 --zone=us-central1-f \
  --ssh-key-file=/Users/cispa/.ssh/id_ed25519
```

On the VM, clone once:

```sh
git clone ~/sc-math-experiments.bundle \
  /mnt/scmath-data/src/sc-math-experiments
```

For later bundle updates on the VM:

```sh
git -C /mnt/scmath-data/src/sc-math-experiments fetch \
  ~/sc-math-experiments.bundle main
git -C /mnt/scmath-data/src/sc-math-experiments merge --ff-only FETCH_HEAD
```

The fast-forward merge fails if VM-local edits conflict with the bundle, so
resolve that before continuing. Once the GitHub remote exists, use normal
`git push` locally and `git pull --ff-only` on the VM instead. Do not put API
credentials or raw private evaluation items into Git.
