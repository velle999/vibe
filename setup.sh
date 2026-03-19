#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Vibe Code Setup ==="

# ── 1. Ensure pip is available ────────────────────────────────────────────────
if ! python3 -m pip --version &>/dev/null; then
    echo "[0/3] Installing python-pip via pacman..."
    sudo pacman -S --noconfirm python-pip
fi

# ── 2. Create venv (Arch blocks system-wide pip installs) ─────────────────────
if [ ! -d ".venv" ]; then
    echo "[1/3] Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# ── 2b. Install CUDA toolkit if missing ───────────────────────────────────────
if ! command -v nvcc &>/dev/null; then
    echo "[1b/3] Installing CUDA toolkit via pacman..."
    sudo pacman -S --noconfirm cuda
    # Add CUDA to PATH for this session
    export PATH="/opt/cuda/bin:$PATH"
    export CUDA_HOME=/opt/cuda
fi

# Ensure CUDA is on PATH even if already installed
export PATH="${CUDA_HOME:-/opt/cuda}/bin:$PATH"
export CUDAToolkit_ROOT="${CUDA_HOME:-/opt/cuda}"

# ── 3. Install llama-cpp-python with CUDA ─────────────────────────────────────
echo "[2/3] Installing llama-cpp-python with CUDA support..."
CMAKE_ARGS="-DGGML_CUDA=on -DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}" \
    pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
pip install rich prompt_toolkit

# ── 4. Download model ─────────────────────────────────────────────────────────
echo "[3/3] Downloading Qwen3.5-9B-Q8_0.gguf (~9.7GB)..."
mkdir -p models
MODEL_PATH="models/Qwen3-8B-Q8_0.gguf"
if [ -f "$MODEL_PATH" ]; then
    echo "Model already exists, skipping download."
else
    curl -L --progress-bar \
        "https://huggingface.co/bartowski/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf" \
        -o "$MODEL_PATH"
fi

echo ""
echo "Setup complete!"
echo "Run with:  source .venv/bin/activate && python main.py"
