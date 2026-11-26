"""
《30天从0转行AI》第 03 课（Day 2 上）完整可运行 PyTorch 实战代码
覆盖：7行手写 LLaMA/DeepSeek 原版同款 RMSNorm、Pre-RMSNorm SwiGLU 残差块、ResNet Bottleneck 与 Roofline 速算！
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 工程关键细节：先转 float32 算均方根防止 BF16 精度丢失
        norm_x = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x.type_as(x) * self.weight


class SwiGLUResBlock(nn.Module):
    def __init__(self, dim: int = 512, hidden_dim: int = 1408):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))


class ResNetBottleneck(nn.Module):
    def __init__(self, c_in: int = 256, c_mid: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_mid, 1, bias=False), nn.BatchNorm2d(c_mid), nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_mid, 3, padding=1, bias=False), nn.BatchNorm2d(c_mid), nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_in, 1, bias=False), nn.BatchNorm2d(c_in),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x))


if __name__ == "__main__":
    x = torch.randn(2, 16, 512)
    blk = SwiGLUResBlock(512, 1408)
    y = blk(x)
    print(f"SwiGLUResBlock 输出 Shape: {y.shape} | 参数量: {sum(p.numel() for p in blk.parameters()):,}")

    img = torch.randn(2, 256, 28, 28)
    bn = ResNetBottleneck(256, 64)
    print(f"ResNetBottleneck 输出 Shape: {bn(img).shape} | 参数量: {sum(p.numel() for p in bn.parameters()):,} (相比直接3x3暴省 8.4x!)")

    print("\n=== GPU 吐字速度物理上限速算器 (Tokens/s ≈ 带宽 / 权重体积) ===")
    for name, bw_gbs, model_gb in [
        ("RTX 4090 + 7B BF16 (14GB)", 1008, 14.0),
        ("RTX 4090 + 7B INT4 (3.5GB)", 1008, 3.5),
        ("H100 SXM + 70B FP8 (70GB)", 3350, 70.0),
        ("Mac M3 Max + 70B INT4 (35GB)", 400, 35.0),
    ]:
        print(f"  {name:<30} -> 理论物理天花板 ≈ {bw_gbs / model_gb:6.1f} Tokens/s")
