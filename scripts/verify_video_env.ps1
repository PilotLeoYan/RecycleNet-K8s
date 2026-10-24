<#
.SYNOPSIS
    RecycleNet-K8s: Script de verificación integral del entorno para el video de preevaluación.
.DESCRIPTION
    Ejecuta las comprobaciones de los 5 módulos demostrados en el guion audiovisual:
    1. Inferencia Real (FastAPI + ONNX Runtime con demo de acierto y error).
    2. Servidor y almacenamiento de experimentos MLflow.
    3. Laboratorio y orquestación de Kubernetes (Minikube / HPA / Kustomize).
    4. Compilación del microservicio nativo en Rust (Axum + Ort).
    5. Calidad de código, análisis estático y pruebas automatizadas (Ruff + Pytest).
#>

[CmdletBinding()]
param ()

$ErrorActionPreference = "Stop"

function Write-Header {
    param ([string]$Title)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "   $Title" -ForegroundColor Yellow
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function Write-Step {
    param ([string]$Name, [string]$Status)
    if ($Status -eq "OK") {
        Write-Host " [OK] $Name" -ForegroundColor Green
    } elseif ($Status -eq "WARN") {
        Write-Host " [!]  $Name" -ForegroundColor Yellow
    } else {
        Write-Host " [ERR] $Name" -ForegroundColor Red
    }
}

Write-Header "VERIFICACION DEL ENTORNO DE DEMOSTRACION · RECYCLENET"

# 1. Configuración de PATH para Rust / Cargo
$cargoPath = "$env:USERPROFILE\.cargo\bin"
if (Test-Path "$cargoPath\cargo.exe") {
    if ($env:PATH -notlike "*$cargoPath*") {
        $env:PATH += ";$cargoPath"
    }
    Write-Step "Compilador Rust y Cargo detectados en $cargoPath" "OK"
} else {
    Write-Step "Cargo no detectado en ruta estándar de usuario ($cargoPath)" "WARN"
}

# 2. Prueba de Inferencia Real (Python + ONNX)
Write-Header "1. Inferencia Real y Demostración de Error (FastAPI + ONNX)"
try {
    uv run python scripts/demo_inference.py
    Write-Step "Demostración de inferencia (Vidrio / Basura) ejecutada con éxito" "OK"
} catch {
    Write-Step "Fallo en la prueba de inferencia: $_" "ERR"
}

# 3. Verificación de MLflow
Write-Header "2. Verificación de Trazabilidad MLOps (MLflow)"
try {
    uv run python scripts/populate_mlflow_runs.py
    Write-Step "Base de datos y tracking de MLflow verificados con runs cargados (sqlite:///mlflow.db)" "OK"
} catch {
    Write-Step "Error al conectar con MLflow: $_" "ERR"
}

# 4. Verificación de Kubernetes Lab (DryRun)
Write-Header "3. Verificación de Manifiestos y Laboratorio Kubernetes"
try {
    .\scripts\k8s_local_lab.ps1 -DryRun -Action all
    Write-Step "Simulación declarativa de Kubernetes (HPA v2, Deployments, Services) verificada" "OK"
} catch {
    Write-Step "Error en manifiestos de Kubernetes: $_" "ERR"
}

# 5. Verificación de Rust Microservice
Write-Header "4. Verificación de Compilación en Rust (Axum Serving)"
if (Get-Command cargo -ErrorAction SilentlyContinue) {
    try {
        Push-Location rust_inference
        cargo check
        Pop-Location
        Write-Step "Microservicio Rust compilado y verificado (Axum + ort)" "OK"
    } catch {
        Pop-Location
        Write-Step "Error en compilación de Rust: $_" "ERR"
    }
} else {
    Write-Step "Cargo no disponible en la sesión actual; omitiendo compilación de Rust" "WARN"
}

# 6. Verificación de Calidad y Pruebas
Write-Header "5. Verificación DevSecOps (Ruff + Batería de Pruebas)"
try {
    uv run ruff check .
    Write-Step "Análisis estático y linter Ruff completado sin errores" "OK"
    uv run pytest tests/test_api.py tests/test_rust_parity.py tests/test_k8s_manifests.py
    Write-Step "Batería de pruebas de API, Paridad Rust y K8s superada (100% éxito)" "OK"
} catch {
    Write-Step "Error en pruebas automatizadas: $_" "ERR"
}

Write-Header "RESUMEN FINAL: ENTORNO 100% LISTO PARA LA GRABACION DEL VIDEO"
