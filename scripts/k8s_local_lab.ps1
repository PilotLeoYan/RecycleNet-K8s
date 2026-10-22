<#
.SYNOPSIS
    RecycleNet-K8s: Local Kubernetes Lab Automation Script for Windows PowerShell.

.DESCRIPTION
    Automates the 5-step local Kubernetes orchestration and HPA lifecycle:
    1. Cluster bootstrap (Minikube with Docker driver, 4 CPUs, 4GB RAM)
    2. Cluster metrics infrastructure (metrics-server addon)
    3. In-cluster image builds directly within Minikube's Docker daemon
    4. Declarative manifest application (Kustomize) and deployment rollout monitoring
    5. Live HPA autoscaling verification (k6 stress load against service endpoints)

.PARAMETER Action
    The lab action to execute:
    - 'start'  : Start Minikube cluster and enable metrics-server
    - 'build'  : Build Python and Rust container images inside Minikube's Docker daemon
    - 'deploy' : Apply declarative manifests (k8s/) and wait for rollout completion
    - 'stress' : Port-forward and run k6 load test to trigger HPA scaling
    - 'status' : Display Minikube, Pods, Deployments, Services, and HPA status
    - 'clean'  : Delete all deployed RecycleNet Kubernetes resources
    - 'all'    : Execute complete 5-step lab workflow sequentially (default)

.PARAMETER DryRun
    Simulate execution and print all underlying commands without executing them.
    No running Minikube or Docker daemon is required when -DryRun is active.

.EXAMPLE
    .\scripts\k8s_local_lab.ps1 -DryRun -Action all
    .\scripts\k8s_local_lab.ps1 -Action start
    .\scripts\k8s_local_lab.ps1 -Action build
    .\scripts\k8s_local_lab.ps1 -Action deploy
    .\scripts\k8s_local_lab.ps1 -Action stress
    .\scripts\k8s_local_lab.ps1 -Action status
    .\scripts\k8s_local_lab.ps1 -Action clean
#>

[CmdletBinding()]
param (
    [Parameter(Position = 0)]
    [ValidateSet('start', 'build', 'deploy', 'stress', 'status', 'clean', 'all')]
    [string]$Action = 'all',

    [Parameter()]
    [switch]$DryRun
)

# Configuration defaults
$ClusterCpus = 4
$ClusterMemory = 4096
$ClusterDriver = 'docker'
$DeployTimeout = '120s'

function Write-LabLog {
    param (
        [string]$Message,
        [string]$Level = 'INFO'
    )
    switch ($Level) {
        'INFO'    { Write-Host "[INFO] $Message" -ForegroundColor Cyan }
        'SUCCESS' { Write-Host "[SUCCESS] $Message" -ForegroundColor Green }
        'WARN'    { Write-Host "[WARN] $Message" -ForegroundColor Yellow }
        'ERROR'   { Write-Host "[ERROR] $Message" -ForegroundColor Red }
        'DRYRUN'  { Write-Host "[DRY-RUN] $Message" -ForegroundColor Yellow }
        'EXEC'    { Write-Host "[EXEC] $Message" -ForegroundColor DarkCyan }
    }
}

function Invoke-LabCommand {
    param (
        [string]$CommandLine,
        [switch]$IgnoreErrors
    )
    if ($DryRun) {
        Write-LabLog -Message $CommandLine -Level 'DRYRUN'
    }
    else {
        Write-LabLog -Message $CommandLine -Level 'EXEC'
        try {
            Invoke-Expression $CommandLine
        }
        catch {
            if (-not $IgnoreErrors) {
                Write-LabLog -Message "Command execution failed: $_" -Level 'ERROR'
                throw $_
            }
            else {
                Write-LabLog -Message "Command exited with non-critical warning: $_" -Level 'WARN'
            }
        }
    }
}

function Start-LocalCluster {
    Write-LabLog -Message "=== Step 1: Initializing Local Minikube Cluster ===" -Level 'INFO'
    Write-LabLog -Message "Cluster specs: driver=$ClusterDriver, cpus=$ClusterCpus, memory=${ClusterMemory}MB" -Level 'INFO'
    Invoke-LabCommand -CommandLine "minikube start --driver=$ClusterDriver --cpus=$ClusterCpus --memory=$ClusterMemory"
    Write-LabLog -Message "Minikube cluster started successfully." -Level 'SUCCESS'
}

function Enable-MetricsServer {
    Write-LabLog -Message "=== Step 2: Enabling Metrics Server Addon ===" -Level 'INFO'
    Write-LabLog -Message "Metrics Server is essential for Horizontal Pod Autoscaler (HPA v2) to poll CPU/memory metrics." -Level 'INFO'
    Invoke-LabCommand -CommandLine "minikube addons enable metrics-server"
    Write-LabLog -Message "Metrics Server addon enabled." -Level 'SUCCESS'
}

function Build-ClusterImages {
    Write-LabLog -Message "=== Step 3: Building Container Images inside Minikube Docker Daemon ===" -Level 'INFO'
    Write-LabLog -Message "Direct build inside Minikube daemon avoids external registry push overhead." -Level 'INFO'

    if ($DryRun) {
        Write-LabLog -Message "minikube -p minikube docker-env --shell powershell | Invoke-Expression" -Level 'DRYRUN'
        Write-LabLog -Message "docker build -t recyclenet-api:local ." -Level 'DRYRUN'
        Write-LabLog -Message "docker build -f Dockerfile.rust -t recyclenet-rust:local ." -Level 'DRYRUN'
    }
    else {
        Write-LabLog -Message "Configuring PowerShell session to use Minikube Docker daemon..." -Level 'INFO'
        & minikube -p minikube docker-env --shell powershell | Invoke-Expression

        Write-LabLog -Message "Building Python FastAPI baseline image (recyclenet-api:local)..." -Level 'INFO'
        docker build -t recyclenet-api:local .

        Write-LabLog -Message "Building Rust Axum high-performance edge image (recyclenet-rust:local)..." -Level 'INFO'
        docker build -f Dockerfile.rust -t recyclenet-rust:local .
    }
    Write-LabLog -Message "All container images built into local Minikube daemon successfully." -Level 'SUCCESS'
}

function Deploy-Manifests {
    Write-LabLog -Message "=== Step 4: Deploying Declarative Manifests & Monitoring Rollout ===" -Level 'INFO'
    Write-LabLog -Message "Applying resources via Kustomize (ConfigMap, Secret, Deployments, Service, HPA, Ingress)..." -Level 'INFO'

    Invoke-LabCommand -CommandLine "kubectl apply -k k8s/"

    Write-LabLog -Message "Waiting for Python deployment rollout ($DeployTimeout)..." -Level 'INFO'
    Invoke-LabCommand -CommandLine "kubectl rollout status deployment/recyclenet-deployment --timeout=$DeployTimeout"

    Write-LabLog -Message "Waiting for Rust deployment rollout ($DeployTimeout)..." -Level 'INFO'
    Invoke-LabCommand -CommandLine "kubectl rollout status deployment/recyclenet-rust-deployment --timeout=$DeployTimeout"

    Write-LabLog -Message "Manifests applied and workloads are fully ready." -Level 'SUCCESS'
}

function Test-HpaScaling {
    Write-LabLog -Message "=== Step 5: Live HPA Autoscaling Stress Test ===" -Level 'INFO'
    Write-LabLog -Message "Exposing service and launching k6 stress benchmark to breach the 70% CPU threshold..." -Level 'INFO'

    if ($DryRun) {
        Write-LabLog -Message "Start-Process -NoNewWindow kubectl -ArgumentList 'port-forward svc/recyclenet-service 8000:8000 8080:8080'" -Level 'DRYRUN'
        Write-LabLog -Message "k6 run benchmarks/k6_load_test.js" -Level 'DRYRUN'
        Write-LabLog -Message "kubectl get hpa recyclenet-hpa" -Level 'DRYRUN'
        Write-LabLog -Message "kubectl get pods -l app=recyclenet" -Level 'DRYRUN'
    }
    else {
        Write-LabLog -Message "Starting background port-forwarding to recyclenet-service (8000:8000, 8080:8080)..." -Level 'INFO'
        $pfProcess = Start-Process -PassThru -NoNewWindow kubectl -ArgumentList "port-forward svc/recyclenet-service 8000:8000 8080:8080"

        try {
            Start-Sleep -Seconds 3

            $k6Installed = Get-Command k6 -ErrorAction SilentlyContinue
            if ($k6Installed) {
                Write-LabLog -Message "Running k6 load test suite against Python endpoint (http://localhost:8000)..." -Level 'INFO'
                $env:TARGET_URL = "http://localhost:8000"
                k6 run benchmarks/k6_load_test.js
            }
            else {
                Write-LabLog -Message "k6 binary not found in PATH." -Level 'WARN'
                Write-LabLog -Message "Executing Python benchmark suite fallback: uv run python -m benchmarks.benchmark_suite..." -Level 'INFO'
                uv run python benchmarks/benchmark_suite.py --iterations 30
            }

            Write-Host ""
            Write-LabLog -Message "=== Autoscaler Scaling Status ===" -Level 'INFO'
            kubectl get hpa recyclenet-hpa
            Write-Host ""
            Write-LabLog -Message "=== Active Pod Replicas ===" -Level 'INFO'
            kubectl get pods -l app=recyclenet -o wide
        }
        finally {
            if ($pfProcess -and -not $pfProcess.HasExited) {
                Write-LabLog -Message "Stopping background port-forward process (PID: $($pfProcess.Id))..." -Level 'INFO'
                Stop-Process -Id $pfProcess.Id -Force -ErrorAction SilentlyContinue
            }
        }
    }
    Write-LabLog -Message "HPA verification completed." -Level 'SUCCESS'
}

function Show-ClusterStatus {
    Write-LabLog -Message "=== RecycleNet Local Kubernetes Cluster Status ===" -Level 'INFO'
    Invoke-LabCommand -CommandLine "minikube status" -IgnoreErrors
    Write-Host ""
    Write-LabLog -Message "=== Cluster Workloads & Networking ===" -Level 'INFO'
    Invoke-LabCommand -CommandLine "kubectl get pods,svc,hpa,ingress -l app=recyclenet -o wide" -IgnoreErrors
    Write-Host ""
    Write-LabLog -Message "=== Workload Resource Consumption (Metrics Server) ===" -Level 'INFO'
    Invoke-LabCommand -CommandLine "kubectl top pods -l app=recyclenet" -IgnoreErrors
}

function Clean-ClusterResources {
    Write-LabLog -Message "=== Cleaning RecycleNet Kubernetes Resources ===" -Level 'INFO'
    Invoke-LabCommand -CommandLine "kubectl delete -k k8s/ --ignore-not-found=true" -IgnoreErrors
    Write-LabLog -Message "RecycleNet workloads and network primitives cleaned up." -Level 'SUCCESS'
    Write-LabLog -Message "To completely stop Minikube, execute: minikube stop" -Level 'INFO'
}

# ------------------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------------------
Write-LabLog -Message "Starting RecycleNet-K8s Local Lab [Action: $Action, DryRun: $DryRun]" -Level 'INFO'

switch ($Action) {
    'start' {
        Start-LocalCluster
        Enable-MetricsServer
    }
    'build' {
        Build-ClusterImages
    }
    'deploy' {
        Deploy-Manifests
    }
    'stress' {
        Test-HpaScaling
    }
    'status' {
        Show-ClusterStatus
    }
    'clean' {
        Clean-ClusterResources
    }
    'all' {
        Start-LocalCluster
        Enable-MetricsServer
        Build-ClusterImages
        Deploy-Manifests
        Test-HpaScaling
        Show-ClusterStatus
    }
}

Write-LabLog -Message "RecycleNet-K8s Local Lab execution finished successfully." -Level 'SUCCESS'
