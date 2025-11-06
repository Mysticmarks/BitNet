from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path
import types

import numpy as np
import pytest

try:  # pragma: no cover - optional dependency
    import torch
    from torch.utils.cpp_extension import load as load_extension
except Exception as exc:  # pragma: no cover - dependency not available
    pytest.skip(f"PyTorch extensions unavailable: {exc}", allow_module_level=True)


KERNEL_DIR = Path(__file__).resolve().parents[1] / "gpu" / "bitnet_kernels"
CUDA_SOURCE = KERNEL_DIR / "bitnet_kernels.cu"

if not CUDA_SOURCE.exists():  # pragma: no cover - repository corruption guard
    pytest.skip("BitNet CUDA sources missing", allow_module_level=True)

_MODULE_CACHE: dict[str, types.ModuleType] = {}
_TEMP_BUILD_DIR: tempfile.TemporaryDirectory[str] | None = None


def _load_bitnet_module() -> types.ModuleType:
    global _TEMP_BUILD_DIR
    if not torch.cuda.is_available():  # pragma: no cover - hardware dependent
        pytest.skip("CUDA runtime not available")

    module = _MODULE_CACHE.get("bitnet")
    if module is not None:
        return module

    if _TEMP_BUILD_DIR is None:
        _TEMP_BUILD_DIR = tempfile.TemporaryDirectory(prefix="bitnet_gemv_")
    build_dir = Path(_TEMP_BUILD_DIR.name)

    binding_code = f"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "bitnet_kernels.h"

void launch_bitlinear(torch::Tensor input,
                      torch::Tensor weight,
                      torch::Tensor output,
                      torch::Tensor act_scale,
                      torch::Tensor weight_scale,
                      int M,
                      int N,
                      int K) {{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA");
    auto stream = at::cuda::getCurrentCUDAStream();
    bitlinear_int8xint2(
        input.data_ptr<int8_t>(),
        weight.data_ptr<int8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(act_scale.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(weight_scale.data_ptr<at::BFloat16>()),
        M,
        N,
        K,
        stream.stream());
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("launch_bitlinear", &launch_bitlinear, "Launch BitNet GEMV kernel");
}}
"""

    binding_path = build_dir / "binding.cpp"
    binding_path.write_text(binding_code, encoding="utf-8")

    try:
        module = load_extension(
            name="bitnet_gemv_runtime",
            sources=[str(binding_path), str(CUDA_SOURCE)],
            extra_include_paths=[str(KERNEL_DIR)],
            extra_cflags=["-O2"],
            extra_cuda_cflags=["-O2"],
            verbose=False,
        )
    except (RuntimeError, OSError) as exc:  # pragma: no cover - compilation failure
        pytest.skip(f"Unable to compile BitNet kernels: {exc}")

    _MODULE_CACHE["bitnet"] = module
    return module


def _pack_2bit(values: np.ndarray) -> np.ndarray:
    rows, cols = values.shape
    assert cols % 4 == 0, "Column count must be divisible by four"
    reshaped = values.reshape(rows, cols // 4, 4)
    packed = (
        ((reshaped[:, :, 0] & 0x3) << 6)
        | ((reshaped[:, :, 1] & 0x3) << 4)
        | ((reshaped[:, :, 2] & 0x3) << 2)
        | (reshaped[:, :, 3] & 0x3)
    )
    return packed.astype(np.uint8)


def _unpack_2bit(packed: np.ndarray, rows: int, cols: int) -> np.ndarray:
    packed = packed.reshape(rows, cols // 4)
    col_indices = np.arange(cols)
    block = col_indices // 4
    shift = 6 - 2 * (col_indices % 4)
    unpacked = (packed[:, block] >> shift) & 0x3
    return unpacked.astype(np.int16)


@pytest.fixture(scope="module")
def gemv_module() -> types.ModuleType:
    return _load_bitnet_module()


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::torch.jit._recursive.WeakScriptModuleProxy")
def test_bitnet_gemv_matches_reference(gemv_module: types.ModuleType, perf_recorder):
    M, N, K = 1, 2560, 2560

    device = torch.device("cuda")
    rng = torch.Generator(device="cpu").manual_seed(1234)
    activations = torch.randint(-128, 127, (K,), dtype=torch.int8, generator=rng).to(device)

    qvalues = torch.randint(0, 4, (N, K), dtype=torch.int32, generator=rng).numpy()
    packed = _pack_2bit(qvalues.astype(np.int16))
    packed_tensor = torch.from_numpy(packed.view(np.int8).reshape(-1)).to(device=device, dtype=torch.int8)

    s = torch.tensor([127.0], dtype=torch.bfloat16, device=device)
    ws = torch.full((N,), 1.0 / 127.0, dtype=torch.bfloat16, device=device)
    output = torch.empty(N, dtype=torch.bfloat16, device=device)

    gemv_module.launch_bitlinear(activations, packed_tensor, output, s, ws, M, N, K)
    torch.cuda.synchronize()

    packed_float = _unpack_2bit(packed.astype(np.int16), N, K)
    dequant_map = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    weights_float = dequant_map[packed_float]

    cpu_activation = activations.cpu().to(torch.float32).numpy()
    cpu_dot = weights_float @ cpu_activation
    cpu_output = (cpu_dot / float(s.cpu().numpy()[0])) * ws.cpu().to(torch.float32).numpy()

    gpu_output = output.detach().cpu().to(torch.float32).numpy()
    max_diff = np.max(np.abs(gpu_output - cpu_output))

    assert output.dtype == torch.bfloat16
    assert max_diff < 1e-2

    iterations = 5
    start = time.perf_counter()
    for _ in range(iterations):
        gemv_module.launch_bitlinear(activations, packed_tensor, output, s, ws, M, N, K)
    torch.cuda.synchronize()
    end = time.perf_counter()
    elapsed = (end - start) / iterations
    tokens_per_second = N / elapsed

    perf_recorder.record("gemv_tokens_per_second", tokens_per_second, unit="tok/s")
    assert math.isfinite(tokens_per_second)
    assert tokens_per_second > 0
