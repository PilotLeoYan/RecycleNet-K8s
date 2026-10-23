#!/usr/bin/env bash
# ==============================================================================
# RecycleNet - Zero-Cost Google Cloud Run Deployment Script
#
# Deploys RecycleNet FastAPI inference service to Cloud Run with strict
# wallet-protection constraints:
#   - min-instances=0  (Scale to zero: $0 cost when idle)
#   - max-instances=1  (Hard cap: prevents runaway instances/billing)
#   - cpu-throttling   (Only billed during active request execution)
# ==============================================================================

set -euo pipefail

# Ensure ~/.local/bin and google-cloud-sdk are in PATH
export PATH="${HOME}/.local/bin:${HOME}/.local/share/google-cloud-sdk/bin:${PATH}"

# ------------------------------------------------------------------------------
# Configuration Defaults
# ------------------------------------------------------------------------------
PROJECT_ID="${GCP_PROJECT_ID:-recyclenet-k8s}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${GCP_SERVICE_NAME:-recyclenet-service}"
REPOSITORY="${GCP_REPOSITORY:-recyclenet-images}"
IMAGE_TAG="${GCP_IMAGE_TAG:-demo-$(date +%s)}"
LOCAL_IMAGE="recyclenet-api:demo"
GAR_HOST="${REGION}-docker.pkg.dev"
IMAGE_URI="${GAR_HOST}/${PROJECT_ID}/${REPOSITORY}/recyclenet-api:${IMAGE_TAG}"

echo "================================================================================"
echo "♻️  RecycleNet GCP Cloud Run Deployment (Zero-Cost / Scale-to-Zero)"
echo "================================================================================"
echo "Project ID:       ${PROJECT_ID}"
echo "Region:           ${REGION}"
echo "Service Name:     ${SERVICE_NAME}"
echo "Artifact Repo:    ${REPOSITORY}"
echo "Target Image URI: ${IMAGE_URI}"
echo "================================================================================"

# ------------------------------------------------------------------------------
# Step 1: Verify Prerequisites
# ------------------------------------------------------------------------------
echo ""
echo "🔍 [Step 1/7] Checking environment and tools..."

if ! command -v gcloud &>/dev/null; then
    echo "❌ Error: 'gcloud' CLI is not found in PATH." >&2
    exit 1
fi

if ! command -v docker &>/dev/null; then
    echo "❌ Error: 'docker' is not installed or not in PATH." >&2
    exit 1
fi

# Verify active account
ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null || true)
if [[ -z "${ACTIVE_ACCOUNT}" ]]; then
    echo "⚠️  No authenticated GCP account detected."
    echo "Please authenticate with your Google account by running:"
    echo "    gcloud auth login"
    echo ""
    echo "Then rerun this script."
    exit 1
fi
echo "✅ Authenticated as: ${ACTIVE_ACCOUNT}"

# Configure active GCP project
gcloud config set project "${PROJECT_ID}" --quiet
echo "✅ Active project set to: ${PROJECT_ID}"

# ------------------------------------------------------------------------------
# Step 2: Enable Required GCP APIs
# ------------------------------------------------------------------------------
echo ""
echo "🔌 [Step 2/7] Enabling Cloud Run & Artifact Registry APIs..."
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    --project="${PROJECT_ID}" --quiet
echo "✅ APIs enabled."

# ------------------------------------------------------------------------------
# Step 3: Ensure Artifact Registry Repository Exists
# ------------------------------------------------------------------------------
echo ""
echo "📦 [Step 3/7] Checking Artifact Registry repository '${REPOSITORY}'..."
if ! gcloud artifacts repositories describe "${REPOSITORY}" --location="${REGION}" --project="${PROJECT_ID}" --quiet &>/dev/null; then
    echo "Creating Docker repository '${REPOSITORY}' in ${REGION}..."
    gcloud artifacts repositories create "${REPOSITORY}" \
        --repository-format=docker \
        --location="${REGION}" \
        --project="${PROJECT_ID}" \
        --description="RecycleNet Docker container images" \
        --quiet
    echo "✅ Created repository '${REPOSITORY}'."
else
    echo "✅ Repository '${REPOSITORY}' already exists."
fi

# ------------------------------------------------------------------------------
# Step 4: Configure Docker Authentication
# ------------------------------------------------------------------------------
echo ""
echo "🔑 [Step 4/7] Configuring Docker credential helper for ${GAR_HOST}..."
gcloud auth configure-docker "${GAR_HOST}" --quiet
echo "✅ Docker authentication configured."

# ------------------------------------------------------------------------------
# Step 5: Build / Tag Docker Image
# ------------------------------------------------------------------------------
echo ""
echo "🐳 [Step 5/7] Preparing Docker image..."
if ! docker image inspect "${LOCAL_IMAGE}" &>/dev/null; then
    echo "Building Docker image ${LOCAL_IMAGE}..."
    docker build -t "${LOCAL_IMAGE}" .
else
    echo "Using existing local image '${LOCAL_IMAGE}'."
fi

docker tag "${LOCAL_IMAGE}" "${IMAGE_URI}"
docker tag "${LOCAL_IMAGE}" "${GAR_HOST}/${PROJECT_ID}/${REPOSITORY}/recyclenet-api:latest"
echo "✅ Tagged: ${IMAGE_URI}"

# ------------------------------------------------------------------------------
# Step 6: Push Container to Artifact Registry
# ------------------------------------------------------------------------------
echo ""
echo "🚀 [Step 6/7] Pushing container image to Artifact Registry..."
docker push "${IMAGE_URI}"
docker push "${GAR_HOST}/${PROJECT_ID}/${REPOSITORY}/recyclenet-api:latest"
echo "✅ Image successfully pushed."

# ------------------------------------------------------------------------------
# Step 7: Deploy to Google Cloud Run (Strict Wallet-Protection Mode)
# ------------------------------------------------------------------------------
echo ""
echo "☁️  [Step 7/7] Deploying to Cloud Run with scale-to-zero..."
gcloud run deploy "${SERVICE_NAME}" \
    --image="${IMAGE_URI}" \
    --region="${REGION}" \
    --platform="managed" \
    --port=8000 \
    --memory=1Gi \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=1 \
    --concurrency=80 \
    --allow-unauthenticated \
    --project="${PROJECT_ID}" \
    --quiet

# Retrieve live URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --platform="managed" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --format="value(status.url)")

echo ""
echo "================================================================================"
echo "🎉 DEPLOYMENT SUCCESSFUL!"
echo "================================================================================"
echo "Public Service URL:  ${SERVICE_URL}"
echo "Interactive Swagger: ${SERVICE_URL}/docs"
echo "Health Check:        ${SERVICE_URL}/health"
echo "Predict Endpoint:    ${SERVICE_URL}/predict/"
echo "================================================================================"
echo ""
echo "🧪 Quick Health Verification:"
curl -s "${SERVICE_URL}/health" || true
echo ""
echo ""
echo "💡 To test with a waste image:"
echo "curl -X POST '${SERVICE_URL}/predict/' \\"
echo "     -F 'file=@path/to/waste_image.jpg'"
echo ""
echo "🗑️ To clean up and delete resources after your demo:"
echo "./scripts/teardown_gcp.sh"
echo "================================================================================"
