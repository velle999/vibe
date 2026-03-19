from pathlib import Path

# Backend: "llama_cpp" or "ollama"
BACKEND = "ollama"

# Paths (llama_cpp backend)
ROOT_DIR = Path(__file__).parent.parent
MODEL_PATH = ROOT_DIR / "models" / "Qwen3-8B-Q8_0.gguf"

# Ollama backend
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen3.5:9b"

# Model settings
N_GPU_LAYERS = -1       # offload all layers to GPU
N_CTX = 32768           # context window
N_THREADS = 8
FLASH_ATTN = True
# Q8_0 KV cache: halves KV VRAM vs F16 default (~2.4GB vs ~4.8GB at 32k)
# Lets 32k context fit on 12GB alongside the 8B Q8_0 weights (~8.5GB)
KV_CACHE_TYPE = 8       # 8 = Q8_0

# Generation settings
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
REPEAT_PENALTY = 1.1
MAX_TOKENS = 8192

# Qwen3 thinking mode: True = show CoT reasoning, False = /no_think
THINKING = True
