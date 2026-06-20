"""code2lora-mlx: a faithful MLX/Apple-Silicon reimplementation of code2lora-lite.

Mirrors the Candle/Rust design:
  repo (.py files) -> MiniLM 768-d embedding -> hypernetwork -> per-layer LoRA
  -> inject into a frozen Qwen2.5-Coder-0.5B -> code completion.
"""

__version__ = "0.1.0"
