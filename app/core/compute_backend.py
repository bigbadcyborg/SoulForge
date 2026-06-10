"""Compute backend detection for GPU vs CPU inference."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import AppConfig


@dataclass(frozen=True)
class ComputeBackend:
    mode: str  # "gpu" | "cpu" | "unknown"
    label: str  # "GPU" | "CPU" | "…"
    detail: str  # human-readable reason for /status


UNKNOWN = ComputeBackend(
    mode="unknown",
    label="…",
    detail="Model not loaded yet",
)


def detect_compute_backend(config: AppConfig) -> ComputeBackend:
    """Determine whether inference is configured for GPU or CPU offload."""
    gpu_layers = config.model.gpu_layers

    if gpu_layers == 0:
        return ComputeBackend(
            mode="cpu",
            label="CPU",
            detail="gpuLayers set to 0 (CPU only)",
        )

    try:
        import llama_cpp

        if not llama_cpp.llama_supports_gpu_offload():
            return ComputeBackend(
                mode="cpu",
                label="CPU",
                detail="llama-cpp-python built without CUDA GPU offload",
            )
    except ImportError:
        return ComputeBackend(
            mode="cpu",
            label="CPU",
            detail="llama-cpp-python not installed",
        )

    layer_desc = "all layers" if gpu_layers < 0 else f"{gpu_layers} layer(s)"
    return ComputeBackend(
        mode="gpu",
        label="GPU",
        detail=f"gpuLayers={gpu_layers} ({layer_desc} on GPU)",
    )
