# Current VM

`sc-math-2xa100` is in project `rg-muandet-15801-1`, zone `us-central1-f`,
with two A100 40 GB GPUs. It uses pinned image
`common-cu129-ubuntu-2404-nvidia-580-v20260909`, OS Login, a 100 GB boot
disk, and a separate 1 TB balanced persistent disk. The latter is ext4,
labelled `scmath-data`, mounted at `/mnt/scmath-data`, and listed in
`/etc/fstab` by UUID.

Verify the VM from a local terminal:

```sh
gcloud compute ssh sc-math-2xa100 --project=rg-muandet-15801-1 \
  --zone=us-central1-f --command='nvidia-smi && df -h /mnt/scmath-data'
```

Stop it after experiments to stop GPU compute billing:

```sh
gcloud compute instances stop sc-math-2xa100 --project=rg-muandet-15801-1 \
  --zone=us-central1-f
```

Stopping releases the GPUs; starting again depends on zone capacity.
Persistent disks remain billable while the VM is stopped. Avoid storing
experiment results only on the boot disk. The old blank
`sc-math-a100-data` disk in `europe-west4-a` is separate and is not attached
to this VM.
