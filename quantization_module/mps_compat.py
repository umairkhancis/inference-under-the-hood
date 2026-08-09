"""Make torch's device-agnostic accelerator APIs usable on MPS (torch 2.12).

llm-compressor drives memory accounting and host-tensor pinning through
`torch.accelerator.*` / `Tensor.pin_memory()`, both of which assume a
CUDA-style caching allocator. On MPS those raise instead of returning
anything:

    torch.accelerator.get_memory_info(0)
    RuntimeError: ... Allocator for mps is not a DeviceAllocator.

    torch.randn(4).pin_memory()
    RuntimeError: Attempted to set the storage of a tensor on device "cpu"
                  to a storage on different device "mps:0".

The `torch.mps.*` equivalents work fine, so redirect to them. Import this
before running any llm-compressor entrypoint. No-op off MPS.
"""

import torch


def install() -> bool:
    """Patch the broken APIs. Returns True if MPS was detected and patched."""
    if not torch.backends.mps.is_available():
        return False

    def get_memory_info(device=None):
        total = torch.mps.recommended_max_memory()
        return (max(total - torch.mps.current_allocated_memory(), 0), total)

    def max_memory_allocated(device=None):
        return torch.mps.driver_allocated_memory()

    def memory_allocated(device=None):
        return torch.mps.current_allocated_memory()

    # MPS keeps no peak counter, so there is nothing to reset.
    def reset_peak_memory_stats(device=None):
        return None

    torch.accelerator.get_memory_info = get_memory_info
    torch.accelerator.max_memory_allocated = max_memory_allocated
    torch.accelerator.memory_allocated = memory_allocated
    torch.accelerator.reset_peak_memory_stats = reset_peak_memory_stats

    # Pinning only exists to enable non-blocking host<->device DMA, which
    # unified memory does not need. Returning the tensor unchanged is safe.
    torch.Tensor.pin_memory = lambda self, device=None: self

    return True
