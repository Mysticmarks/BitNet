import unittest

try:
    import torch
    from torch.utils.cpp_extension import load_inline
except Exception as exc:  # pragma: no cover - dependency not available
    raise unittest.SkipTest(f"PyTorch extensions unavailable: {exc}")


CUDA_SRC = r"""
extern "C" __global__ void add_kernel(const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float* __restrict__ out,
                                      int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    out[idx] = a[idx] + b[idx];
}
"""

MODULE = None


def _load_module():
    global MODULE
    if MODULE is None:
        MODULE = load_inline(
            name="bitnet_gpu_add",
            cpp_sources="",
            cuda_sources=CUDA_SRC,
            functions=["add_kernel"],
            verbose=False,
        )
    return MODULE


class GpuKernelUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA runtime not available")
        self.module = _load_module()

    def test_add_kernel_matches_cpu(self):
        device = torch.device("cuda")
        n = 64
        a = torch.arange(0, n, dtype=torch.float32, device=device)
        b = torch.arange(n, 2 * n, dtype=torch.float32, device=device)
        out = torch.empty_like(a)

        threads = 32
        blocks = (n + threads - 1) // threads
        self.module.add_kernel(
            (blocks,),
            (threads,),
            (a.data_ptr(), b.data_ptr(), out.data_ptr(), n),
        )
        torch.cuda.synchronize()
        expected = (torch.arange(0, n, dtype=torch.float32) + torch.arange(n, 2 * n, dtype=torch.float32)).to(device)
        self.assertTrue(torch.allclose(out, expected))


if __name__ == "__main__":
    unittest.main()
