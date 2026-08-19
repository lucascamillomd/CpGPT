"""Legacy (torch<=2.6) RMSNorm numerics for bit-reproducible mixed-precision inference.

CpGPT checkpoints released with the manuscript were trained and evaluated under
``precision="16-mixed"`` on torch<=2.6.0. Two later PyTorch changes silently alter
mixed-precision predictions:

1. torch 2.7 (pytorch/pytorch#147203) changed how ``F.rms_norm`` handles half-precision
   inputs under autocast. With ``eps=None`` (CpGPT's default), torch<=2.6 resolved eps
   from the *input* dtype (fp16 -> ~9.8e-4) whereas torch>=2.7 upcasts first
   (fp32 -> ~1.2e-7). CpGPT activations have small mean-squares, so this shifts every
   norm layer by 1-2.5% and compounds into multi-year prediction changes
   (e.g. ~15 years on CpGPTGrimAge3).
2. torch 2.13 changed the CUDA fp32 reduction order of ``mean(dim=-1)``, perturbing
   norm outputs at the last bit.

This module restores the exact torch-2.6 CUDA semantics with platform-independent,
IEEE-exact elementwise ops:

- fp16/bf16 inputs: normalize in fp32 with the *fp16* eps, round to half mid-op,
  multiply by the fp32 weight, return fp32 (the torch<=2.6 cast chain).
- fp32 inputs (the transformer residual stream under autocast): normalize in fp32
  with the *fp32* eps, matching the torch<=2.6 native path.
- the ``mean(-1)`` reduction is replicated with a fixed summation order (32 strided
  serial lanes, then an ascending adjacent-pairs tree) that bit-matches the
  torch<=2.6 CUDA kernel.

With this shim active, CUDA inference under 16-mixed autocast is bitwise identical
to torch 2.6.0 outputs on torch 2.6-2.13 (verified on RTX 4090 / sm89 for the
released small-architecture checkpoints across multiple input shapes). Non-CUDA
devices cannot match CUDA bit-for-bit (GEMM/attention accumulation orders are
hardware-specific) but land within the pre-existing cross-device noise
(<~0.1 years on age-scale outputs).

Only ``nn.RMSNorm`` modules with ``eps=None`` running under autocast are affected;
full-precision inference and explicitly-set eps values fall through to native torch.
"""

import torch
from torch import nn

_FP16_EPS = torch.finfo(torch.float16).eps  # 9.765625e-04
_FP32_EPS = torch.finfo(torch.float32).eps  # 1.1920928955078125e-07

_native_rmsnorm_forward = nn.RMSNorm.forward
_installed = False


def _legacy_meansq(xf: torch.Tensor) -> torch.Tensor:
    """Mean of squares over the last dim, bit-matching the torch<=2.6 CUDA kernel.

    Fixed order: lane ``l`` serially sums strided elements ``x[l], x[l+32], ...``;
    the 32 lane partials are combined with an ascending adjacent-pairs tree; the
    division by d is an exact scale for power-of-two d.
    """
    d = xf.shape[-1]
    v = xf * xf
    if d % 32 != 0:
        # The fixed-order tree below assumes 32 lanes (all CpGPT architectures use
        # power-of-two dims). Other dims get the native mean: correct, though not
        # bit-matched to any legacy kernel.
        return v.mean(-1, keepdim=True)
    acc = v[..., 0:32]
    for k in range(1, d // 32):
        acc = acc + v[..., 32 * k : 32 * (k + 1)]
    while acc.shape[-1] > 1:
        acc = acc[..., 0::2] + acc[..., 1::2]
    return acc * (1.0 / d) if (d & (d - 1)) == 0 else acc / d


def _legacy_rmsnorm_forward(self: nn.RMSNorm, x: torch.Tensor) -> torch.Tensor:
    """torch<=2.6 autocast RMSNorm semantics; native behavior otherwise."""
    if self.eps is None and torch.is_autocast_enabled(x.device.type):
        if x.dtype in (torch.float16, torch.bfloat16):
            xf = x.float()
            y = (xf * torch.rsqrt(_legacy_meansq(xf) + _FP16_EPS)).to(x.dtype).float()
            return y * self.weight
        if x.dtype == torch.float32:
            return x * torch.rsqrt(_legacy_meansq(x) + _FP32_EPS) * self.weight
    return _native_rmsnorm_forward(self, x)


def enable_legacy_numerics() -> None:
    """Activate torch<=2.6 RMSNorm numerics process-wide (idempotent).

    Called automatically by ``CpGPTInferencer`` and ``CpGPTTrainer`` so that
    mixed-precision predictions from released checkpoints stay bit-identical to
    the torch 2.6 CUDA outputs they were published with.
    """
    global _installed
    if not _installed:
        nn.RMSNorm.forward = _legacy_rmsnorm_forward
        _installed = True


def disable_legacy_numerics() -> None:
    """Restore native torch RMSNorm numerics (breaks legacy reproducibility)."""
    global _installed
    if _installed:
        nn.RMSNorm.forward = _native_rmsnorm_forward
        _installed = False
