"""Compact Diffusion Policy (Chi et al. 2023): ResNet18+SpatialSoftmax encoders, 1D conditional UNet, DDPM/DDIM."""
import math
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from diffusers import DDPMScheduler, DDIMScheduler


def resnet18_gn():
    m = torchvision.models.resnet18(weights=None, norm_layer=lambda c: nn.GroupNorm(c // 16, c))
    return nn.Sequential(*list(m.children())[:-2])  # B,512,h,w


class SpatialSoftmax(nn.Module):
    def __init__(self, c, k=32):
        super().__init__()
        self.proj = nn.Conv2d(c, k, 1)

    def forward(self, x):
        x = self.proj(x); B, K, H, W = x.shape
        att = F.softmax(x.flatten(2), -1)
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device), torch.linspace(-1, 1, W, device=x.device), indexing="ij")
        return torch.cat([(att * xs.flatten()).sum(-1), (att * ys.flatten()).sum(-1)], -1)  # B,2K


class ImageEncoder(nn.Module):
    def __init__(self, out=128, crop=116):
        super().__init__()
        self.crop = crop
        self.net = nn.Sequential(resnet18_gn(), SpatialSoftmax(512, 32), nn.Linear(64, out), nn.ReLU())

    def forward(self, x):  # x uint8 B,H,W,3
        x = x.permute(0, 3, 1, 2).float() / 255.0
        H = x.shape[-1]; c = self.crop
        if self.training:
            i, j = torch.randint(0, H - c + 1, (2,))
            x = x[..., i:i + c, j:j + c]
        else:
            o = (H - c) // 2; x = x[..., o:o + c, o:o + c]
        x = (x - 0.5) / 0.25
        return self.net(x)


class SinPosEmb(nn.Module):
    def __init__(self, d):
        super().__init__(); self.d = d

    def forward(self, t):
        h = self.d // 2; f = torch.exp(torch.arange(h, device=t.device) * -(math.log(10000) / (h - 1)))
        e = t.float()[:, None] * f[None]
        return torch.cat([e.sin(), e.cos()], -1)


class CondRes(nn.Module):
    def __init__(self, i, o, cdim, k=5):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.GroupNorm(8, o), nn.Mish())
        self.b2 = nn.Sequential(nn.Conv1d(o, o, k, padding=k // 2), nn.GroupNorm(8, o), nn.Mish())
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cdim, 2 * o))
        self.res = nn.Conv1d(i, o, 1) if i != o else nn.Identity()

    def forward(self, x, c):
        h = self.b1(x); s, b = self.film(c)[..., None].chunk(2, 1)
        return self.b2(h * (1 + s) + b) + self.res(x)


class UNet1D(nn.Module):
    def __init__(self, adim, cdim, dims=(256, 512, 1024)):
        super().__init__()
        self.temb = nn.Sequential(SinPosEmb(128), nn.Linear(128, 512), nn.Mish(), nn.Linear(512, 128))
        cd = cdim + 128; ch = [adim] + list(dims)
        self.down = nn.ModuleList(); self.up = nn.ModuleList()
        for n, (a, b) in enumerate(zip(ch[:-1], ch[1:])):
            last = n == len(dims) - 1
            self.down.append(nn.ModuleList([CondRes(a, b, cd), CondRes(b, b, cd),
                                            nn.Conv1d(b, b, 3, 2, 1) if not last else nn.Identity()]))
        self.mid = nn.ModuleList([CondRes(dims[-1], dims[-1], cd), CondRes(dims[-1], dims[-1], cd)])
        for n, (a, b) in enumerate(zip(reversed(ch[2:]), reversed(ch[1:-1]))):
            self.up.append(nn.ModuleList([CondRes(a * 2, b, cd), CondRes(b, b, cd), nn.ConvTranspose1d(b, b, 4, 2, 1)]))
        self.out = nn.Sequential(nn.Conv1d(dims[0], dims[0], 5, padding=2), nn.GroupNorm(8, dims[0]), nn.Mish(), nn.Conv1d(dims[0], adim, 1))

    def forward(self, x, t, cond):  # x B,T,A
        x = x.transpose(1, 2); c = torch.cat([self.temb(t), cond], -1); hs = []
        for r1, r2, ds in self.down:
            x = r2(r1(x, c), c); hs.append(x); x = ds(x)
        for m in self.mid: x = m(x, c)
        for r1, r2, us in self.up:
            x = torch.cat([x, hs.pop()], 1); x = us(r2(r1(x, c), c))
        return self.out(x).transpose(1, 2)


class DiffusionPolicy(nn.Module):
    def __init__(self, stats, n_obs=2, horizon=16, n_act=8, adim=7, pdim=8, n_tasks=0, cams=("agent", "wrist"), text_dim=0, crop=116):
        super().__init__()
        self.n_obs, self.horizon, self.n_act, self.cams = n_obs, horizon, n_act, cams
        self.enc = nn.ModuleDict({c: ImageEncoder(crop=crop) for c in cams})
        self.task_emb = nn.Embedding(n_tasks, 64) if n_tasks else None
        self.text_proj = nn.Sequential(nn.Linear(text_dim, 128), nn.Mish(), nn.Linear(128, 64)) if text_dim else None
        cdim = n_obs * (128 * len(cams) + pdim) + (64 if n_tasks else 0) + (64 if text_dim else 0)
        self.unet = UNet1D(adim, cdim)
        for k, v in stats.items(): self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))
        self.train_sched = DDPMScheduler(100, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", clip_sample=True)
        self.infer_sched = DDIMScheduler(100, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", clip_sample=True)

    def norm(self, x, k): return 2 * (x - getattr(self, k + "_min")) / (getattr(self, k + "_max") - getattr(self, k + "_min") + 1e-6) - 1
    def unnorm(self, x, k): return (x + 1) / 2 * (getattr(self, k + "_max") - getattr(self, k + "_min") + 1e-6) + getattr(self, k + "_min")

    def cond(self, obs):  # obs[cam]: B,To,H,W,3 uint8 ; obs['proprio']: B,To,P ; obs['tid']: B
        B = obs["proprio"].shape[0]
        f = [self.enc[c](obs[c].flatten(0, 1)).view(B, self.n_obs, -1) for c in self.cams]
        f = torch.cat(f + [self.norm(obs["proprio"], "proprio")], -1).flatten(1)
        if self.task_emb is not None: f = torch.cat([f, self.task_emb(obs["tid"])], -1)
        if self.text_proj is not None: f = torch.cat([f, self.text_proj(obs["text"].float())], -1)
        return f

    def loss(self, obs, action):
        c = self.cond(obs); a = self.norm(action, "action")
        noise = torch.randn_like(a)
        t = torch.randint(0, 100, (a.shape[0],), device=a.device)
        return F.mse_loss(self.unet(self.train_sched.add_noise(a, noise, t), t, c), noise)

    @torch.no_grad()
    def act(self, obs, steps=10):
        c = self.cond(obs); B = c.shape[0]
        a = torch.randn(B, self.horizon, self.action_min.shape[0], device=c.device)
        self.infer_sched.set_timesteps(steps)
        for t in self.infer_sched.timesteps:
            a = self.infer_sched.step(self.unet(a, t.expand(B).to(c.device), c), t, a).prev_sample
        a = self.unnorm(a, "action")
        s = self.n_obs - 1
        return a[:, s:s + self.n_act]
