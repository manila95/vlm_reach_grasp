"""Image augmentations for photometric / texture invariance of the local skill (docs/PLAN_perceptual_robustness.md, S1/S2).

Inputs are uint8 tensors (B, T, H, W, 3) on the GPU; one draw per sample, shared by its T observation frames.
Ranges are generic (chosen before any perturbed-scene result), not fit to any benchmark's perturbations.
  photo: brightness x[0.35, 1.6], contrast [0.6, 1.4], saturation [0.4, 1.6], per-channel tint [0.8, 1.2],
         gamma [0.6, 1.7] (log-uniform), Gaussian noise sigma [0, 0.03]; applied to 80% of samples
  conv:  RandConv (Xu et al. 2021): a random 3x3 conv, mixed with the image by alpha ~ U(0, 1); applied to 50% of samples
"""
import torch
import torch.nn.functional as F


def _u(n, lo, hi, dev): return torch.empty(n, device=dev).uniform_(lo, hi)


def photometric(x, p=0.8):
    B, T, H, W, _ = x.shape; dev = x.device
    y = x.float() / 255.0
    on = (torch.rand(B, device=dev) < p).float().view(B, 1, 1, 1, 1)
    v = lambda t: t.view(B, 1, 1, 1, -1)
    z = y * v(_u(B, 0.35, 1.6, dev))                                                     # brightness
    m = z.mean(dim=(1, 2, 3, 4), keepdim=True); z = (z - m) * v(_u(B, 0.6, 1.4, dev)) + m  # contrast
    g = z.mean(-1, keepdim=True); z = g + (z - g) * v(_u(B, 0.4, 1.6, dev))              # saturation
    z = z * torch.empty(B, 1, 1, 1, 3, device=dev).uniform_(0.8, 1.2)                     # tint
    z = z.clamp(0, 1) ** v(torch.exp(_u(B, -0.511, 0.531, dev)))                          # gamma in [0.6, 1.7]
    z = z + torch.randn_like(z) * v(_u(B, 0.0, 0.03, dev))                                # sensor noise
    y = on * z.clamp(0, 1) + (1 - on) * y
    return (y * 255).round().to(torch.uint8)


def randconv(x, p=0.5):
    B, T, H, W, C = x.shape; dev = x.device
    y = x.float().div(255).permute(0, 1, 4, 2, 3).reshape(1, B * T * C, H, W)
    w = torch.randn(B, 3, 3, 3, 3, device=dev) / (27 ** 0.5)                             # per-sample conv (out, in, 3, 3)
    w = w.repeat_interleave(T, 0).reshape(B * T * 3, 3, 3, 3)
    z = F.conv2d(y, w, padding=1, groups=B * T).reshape(B, T, 3, H, W)
    lo = z.amin(dim=(1, 2, 3, 4), keepdim=True); hi = z.amax(dim=(1, 2, 3, 4), keepdim=True)
    z = (z - lo) / (hi - lo + 1e-6)                                                       # back to [0, 1]
    y = y.reshape(B, T, 3, H, W); a = _u(B, 0, 1, dev).view(B, 1, 1, 1, 1)
    on = (torch.rand(B, device=dev) < p).float().view(B, 1, 1, 1, 1)
    out = on * (a * y + (1 - a) * z) + (1 - on) * y
    return (out.permute(0, 1, 3, 4, 2).clamp(0, 1) * 255).round().to(torch.uint8)


def augment(x, kind):
    if "photo" in kind: x = photometric(x)
    if "conv" in kind: x = randconv(x)
    return x
