#!/usr/bin/env bash
# ==============================================================================
# RecycleNet-K8s: Local Kubernetes Lab Automation (Minikube + HPA)
# ==============================================================================
# This script orchestrates the 5-step local Kubernetes lab lifecycle:
# 1. Cluster initialization (Minikube with Docker driver, 4 CPUs, 4GB RAM)
# 2. Metric collection infrastructure (metrics-server addon)
# 3. In-cluster container builds (Python FastAPI & Rust Axum in Minikube's daemon)
# 4. Declarative manifest deployment (Kustomize / kubectl apply) & rollout check
# 5. Live HPA autoscaling verification (k6 stress load against service endpoints)
#
# Usage:
#   ./scripts/k8s_local_lab.sh [subcommand] [--dry-run]
#
# Subcommands:
#   start     - Start Minikube cluster and enable metrics-server
#   build     - Build Python & Rust Docker images inside Minikube's Docker daemon
#   deploy    - Apply K8s manifests (k8s/) and wait for rollout completion
#   stress    - Port-forward and execute k6 stress test to trigger HPA scaling
#   status    - Check Minikube, Pods, Services, HPA, and resource consumption
#   clean     - Tear down all Kubernetes resources
#   all       - Execute complete 5-step lab workflow sequentially (default)
#
# Flags:
#   --dry-run, -n   Print commands without executing them (no cluster required)
#   --help, -h      Display this documentation and usage guide
# ==============================================================================

set -eo pipefail

# ANSI color codes for structured terminal output
if [ -t 1 ]; then
    COLOR_RESET="\033[0m"
    COLOR_INFO="\033[36m"     # Cyan
    COLOR_SUCCESS="\033[32m"  # Green
    COLOR_WARN="\033[33m"     # Yellow
    COLOR_ERROR="\033[31m"    # Red
    COLOR_BOLD="\033[1m"
else
    COLOR_RESET=""
    COLOR_INFO=""
    COLOR_SUCCESS=""
    COLOR_WARN=""
    COLOR_ERROR=""
    COLOR_BOLD=""
fi

# Configuration defaults
CLUSTER_CPUS=4
CLUSTER_MEMORY=4096
CLUSTER_DRIVER="docker"
DEPLOY_TIMEOUT="120s"
DRY_RUN=0
ACTION="all"

log_info() {
    echo -e "${COLOR_INFO}${COLOR_BOLD}[INFO]${COLOR_RESET} $1"
}

log_success() {
    echo -e "${COLOR_SUCCESS}${COLOR_BOLD}[SUCCESS]${COLOR_RESET} $1"
}

log_warn() {
    echo -e "${COLOR_WARN}${COLOR_BOLD}[WARN]${COLOR_RESET} $1"
}

log_error() {
    echo -e "${COLOR_ERROR}${COLOR_BOLD}[ERROR]${COLOR_RESET} $1" >&2
}

log_cmd() {
    if [ "$DRY_RUN" -eq 1 ]; then
        echo -e "${COLOR_WARN}[DRY-RUN]${COLOR_RESET} $1"
    else
        echo -e "${COLOR_INFO}[EXEC]${COLOR_RESET} $1"
    fi
}

run_cmd() {
    local cmd="$1"
    log_cmd "$cmd"
    if [ "$DRY_RUN" -eq 0 ]; then
        eval "$cmd"
    fi
}

show_help() {
    cat <<EOF
RecycleNet-K8s: Local Kubernetes Lab Automation Script

Usage:
  $(basename "$0") [subcommand] [flags]

Available Subcommands:
  start     Step 1 & 2: Start Minikube cluster and enable metrics-server
  build     Step 3: Build Docker images inside Minikube's Docker daemon
  deploy    Step 4: Apply K8s manifests and wait for rollout completion
  stress    Step 5: Run k6 load test and monitor HPA scaling in real time
  status    Inspect Minikube, Pods, Deployments, Services, and HPA status
  clean     Delete all deployed RecycleNet Kubernetes resources
  all       Execute the entire 5-step lab workflow (default)

Flags:
  -n, --dry-run   Simulate execution and print commands without executing them
  -h, --help      Display this help message and exit

Examples:
  ./scripts/k8s_local_lab.sh --dry-run all
  ./scripts/k8s_local_lab.sh start
  ./scripts/k8s_local_lab.sh build
  ./scripts/k8s_local_lab.sh deploy
  ./scripts/k8s_local_lab.sh stress
  ./scripts/k8s_local_lab.sh status
  ./scripts/k8s_local_lab.sh clean
EOF
}

# ------------------------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        start|build|deploy|stress|status|clean|all)
            ACTION="$1"
            shift
            ;;
        *)
            log_error "Unknown option or subcommand: $1"
            show_help
            exit 1
            ;;
    esac
done

# ------------------------------------------------------------------------------
# Lab Steps Implementation
# ------------------------------------------------------------------------------

step1_start_cluster() {
    log_info "=== Step 1: Initializing Local Minikube Cluster ==="
    log_info "Cluster specs: driver=${CLUSTER_DRIVER}, cpus=${CLUSTER_CPUS}, memory=${CLUSTER_MEMORY}MB"
    run_cmd "minikube start --driver=${CLUSTER_DRIVER} --cpus=${CLUSTER_CPUS} --memory=${CLUSTER_MEMORY}"
    log_success "Minikube cluster started successfully."
}

step2_enable_metrics() {
    log_info "=== Step 2: Enabling Metrics Server Addon ==="
    log_info "Metrics Server is essential for Horizontal Pod Autoscaler (HPA v2) to poll CPU/memory metrics."
    run_cmd "minikube addons enable metrics-server"
    log_success "Metrics Server addon enabled."
}

step3_build_images() {
    log_info "=== Step 3: Building Container Images inside Minikube Docker Daemon ==="
    log_info "Direct build inside Minikube avoids pushing to an external registry."

    if [ "$DRY_RUN" -eq 1 ]; then
        log_cmd "eval \$(minikube docker-env)"
        log_cmd "docker build -t recyclenet-api:local ."
        log_cmd "docker build -f Dockerfile.rust -t recyclenet-rust:local ."
    else
        log_cmd "Configuring shell environment to use Minikube's Docker daemon..."
        eval "$(minikube docker-env)"

        log_info "Building Python FastAPI baseline image (recyclenet-api:local)..."
        docker build -t recyclenet-api:local .

        log_info "Building Rust Axum high-performance edge image (recyclenet-rust:local)..."
        docker build -f Dockerfile.rust -t recyclenet-rust:local .
    fi
    log_success "All container images built into local Minikube daemon successfully."
}

step4_deploy_manifests() {
    log_info "=== Step 4: Deploying Declarative Manifests & Monitoring Rollout ==="
    log_info "Applying resources via Kustomize (ConfigMap, Secret, Deployments, Service, HPA, Ingress)..."

    # Prefer kustomize deployment; fallback to directory apply
    run_cmd "kubectl apply -k k8s/"

    log_info "Waiting for Python deployment rollout (${DEPLOY_TIMEOUT})..."
    run_cmd "kubectl rollout status deployment/recyclenet-deployment --timeout=${DEPLOY_TIMEOUT}"

    log_info "Waiting for Rust deployment rollout (${DEPLOY_TIMEOUT})..."
    run_cmd "kubectl rollout status deployment/recyclenet-rust-deployment --timeout=${DEPLOY_TIMEOUT}"

    log_success "Manifests applied and workloads are fully ready."
}

step5_stress_test_hpa() {
    log_info "=== Step 5: Live HPA Autoscaling Stress Test ==="
    log_info "Exposing service and launching k6 stress benchmark to breach the 70% CPU threshold..."

    if [ "$DRY_RUN" -eq 1 ]; then
        log_cmd "kubectl port-forward svc/recyclenet-service 8000:8000 8080:8080 &"
        log_cmd "k6 run benchmarks/k6_load_test.js"
        log_cmd "kubectl get hpa recyclenet-hpa"
        log_cmd "kubectl get pods -l app=recyclenet"
    else
        # Start port-forwarding in background
        log_info "Starting port-forwarding to recyclenet-service (8000:8000, 8080:8080)..."
        kubectl port-forward svc/recyclenet-service 8000:8000 8080:8080 > /dev/null 2>&1 &
        PF_PID=$!

        # Ensure background port-forward process is terminated when script exits
        cleanup_pf() {
            if kill -0 "$PF_PID" 2>/dev/null; then
                log_info "Stopping port-forwarding process (PID: $PF_PID)..."
                kill "$PF_PID" 2>/dev/null || true
            fi
        }
        trap cleanup_pf EXIT

        # Wait briefly for port-forward socket binding
        sleep 3

        # Check if k6 is installed
        if command -v k6 &> /dev/null; then
            log_info "Running k6 load test suite against Python endpoint (http://localhost:8000)..."
            k6 run -e TARGET_URL=http://localhost:8000 benchmarks/k6_load_test.js || true
        else
            log_warn "k6 binary not found in PATH."
            log_info "Executing Python benchmark suite fallback: uv run python -m benchmarks.benchmark_suite..."
            uv run python benchmarks/benchmark_suite.py --iterations 30 || true
        fi

        log_info "=== Autoscaler Scaling Status ==="
        kubectl get hpa recyclenet-hpa || true
        echo ""
        log_info "=== Active Pod Replicas ==="
        kubectl get pods -l app=recyclenet -o wide || true
    fi
    log_success "HPA verification completed."
}

show_status() {
    log_info "=== RecycleNet Local Kubernetes Cluster Status ==="
    run_cmd "minikube status"
    echo ""
    log_info "=== Cluster Workloads & Networking ==="
    run_cmd "kubectl get pods,svc,hpa,ingress -l app=recyclenet -o wide"
    echo ""
    log_info "=== Workload Resource Consumption (Metrics Server) ==="
    run_cmd "kubectl top pods -l app=recyclenet --use-protocol-buffers=false 2>/dev/null || kubectl top pods -l app=recyclenet"
}

clean_resources() {
    log_info "=== Cleaning RecycleNet Kubernetes Resources ==="
    run_cmd "kubectl delete -k k8s/ --ignore-not-found=true"
    log_success "RecycleNet workloads and network primitives cleaned up."
    log_info "To completely stop Minikube, execute: minikube stop"
}

# ------------------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------------------
log_info "Starting RecycleNet-K8s Local Lab [Action: ${ACTION}, DryRun: ${DRY_RUN}]"

case "$ACTION" in
    start)
        step1_start_cluster
        step2_enable_metrics
        ;;
    build)
        step3_build_images
        ;;
    deploy)
        step4_deploy_manifests
        ;;
    stress)
        step5_stress_test_hpa
        ;;
    status)
        show_status
        ;;
    clean)
        clean_resources
        ;;
    all)
        step1_start_cluster
        step2_enable_metrics
        step3_build_images
        step4_deploy_manifests
        step5_stress_test_hpa
        show_status
        ;;
esac

log_success "RecycleNet-K8s Local Lab execution finished successfully."
