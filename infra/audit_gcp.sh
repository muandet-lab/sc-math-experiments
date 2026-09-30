#!/usr/bin/env bash
# Read-only readiness check. Run after `gcloud auth login` on a trusted terminal.
set -euo pipefail

project="${1:?usage: audit_gcp.sh PROJECT_ID [ZONE]}"
zone="${2:-us-central1-f}"
region="${zone%-*}"

echo "Project: ${project}; zone: ${zone}"
gcloud auth list --filter='status:ACTIVE' --format='value(account)'
gcloud services list --enabled --project="${project}" \
  --filter='config.name:(compute.googleapis.com OR storage.googleapis.com OR aiplatform.googleapis.com)' \
  --format='value(config.name)'
gcloud compute instances list --project="${project}" \
  --format='table(name,zone,machineType,status)'
gcloud compute regions describe "${region}" --project="${project}" \
  --format='table(quotas.metric,quotas.limit,quotas.usage)'
for machine in a2-highgpu-2g a2-highgpu-1g; do
  gcloud compute machine-types describe "${machine}" --zone="${zone}" \
    --project="${project}" --format='table(name,guestCpus,memoryMb,accelerators)'
done
gcloud compute images describe-from-family common-cu129-ubuntu-2404-nvidia-580 \
  --project=deeplearning-platform-release --format='value(name)'
