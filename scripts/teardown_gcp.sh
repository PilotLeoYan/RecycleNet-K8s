#!/usr/bin/env bash
# ==============================================================================
# RecycleNet - GCP Resource Teardown Script
#
# Cleans up Cloud Run service and Artifact Registry repository to guarantee
# zero ongoing storage or cloud footprint after the demo.
# ==============================================================================

set -euo pipefail

# Ensure ~/.local/bin and google-cloud-sdk are in PATH
export PATH="${HOME}/.local/bin:${HOME}/.local/share/google-cloud-sdk/bin:${PATH}"

PROJECT_ID="${GCP_PROJECT_ID:-recyclenet-k8s}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${GCP_SERVICE_NAME:-recyclenet-service}"
REPOSITORY="${GCP_REPOSITORY:-recyclenet-images}"

echo "================================================================================"
echo "🗑️  RecycleNet GCP Resource Teardown"
echo "================================================================================"
echo "Project ID:    ${PROJECT_ID}"
echo "Region:        ${REGION}"
echo "Service Name:  ${SERVICE_NAME}"
echo "Artifact Repo: ${REPOSITORY}"
echo "================================================================================"

echo ""
echo "Deleting Cloud Run service '${SERVICE_NAME}'..."
gcloud run services delete "${SERVICE_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --quiet 2>/dev/null || echo "Service '${SERVICE_NAME}' already deleted or not found."

echo ""
echo "Deleting Artifact Registry repository '${REPOSITORY}'..."
gcloud artifacts repositories delete "${REPOSITORY}" \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --quiet 2>/dev/null || echo "Repository '${REPOSITORY}' already deleted or not found."

echo ""
echo "✅ All resources cleaned up. Your GCP bill will remain \$0.00."
