"""
《30天从0转行AI》第 02 课（Day 1 下）完整可运行 PyTorch 实战代码
覆盖：纯手写矩阵反向传播验算、8行白板手撕 AdamW、自动区分 Decay/No-Decay 参数分组！
"""
import torch
import torch.nn as nn


def demo_manual_backprop():
    print("=== 1. 纯手写线性层矩阵反向传播 vs PyTorch Autograd ===")
    torch.manual_seed(0)
    X = torch.randn(4, 8)
    W = torch.randn(8, 16, requires_grad=True)
    Y = X @ W
    loss = (Y ** 2).mean()
    loss.backward()

    # 链式法则解析解
    dY = (2.0 * Y) / Y.numel()
    dW_manual = X.T @ dY
    assert torch.allclose(W.grad, dW_manual, atol=1e-6)
    print(f"最大逐位误差: {(W.grad - dW_manual).abs().max().item():.2e} -> 完美吻合!")


def adamw_step(p, grad, m, v, t, lr=1e-3, wd=0.01, b1=0.9, b2=0.999, eps=1e-8):
    with torch.no_grad():
        p.mul_(1.0 - lr * wd)
        m.mul_(b1).add_(grad, alpha=1.0 - b1)
        v.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
        m_hat = m / (1.0 - b1 ** t)
        v_hat = v / (1.0 - b2 ** t)
        p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)


def demo_handwrite_adamw():
    print("\n=== 2. 白板手撕 AdamW vs 官方 torch.optim.AdamW ===")
    torch.manual_seed(42)
    p1 = nn.Parameter(torch.randn(4, 4))
    p2 = nn.Parameter(p1.detach().clone())
    g = torch.randn(4, 4)

    opt = torch.optim.AdamW([p1], lr=1e-2, weight_decay=0.1)
    m, v = torch.zeros_like(p2), torch.zeros_like(p2)

    for step in range(1, 6):
        p1.grad = g.clone()
        opt.step()
        adamw_step(p2, g, m, v, step, lr=1e-2, wd=0.1)

    assert torch.allclose(p1, p2, atol=1e-6)
    print(f"连续 5 步后与官方 AdamW 最大误差: {(p1 - p2).abs().max().item():.2e} -> 100% 精确一致!")


def demo_param_groups():
    print("\n=== 3. 大模型标配：2D 权重衰减 + 1D Bias/Norm 豁免 ===")
    model = nn.Sequential(nn.Linear(64, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 10))
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    print(f"Decay 参数张量数: {len(decay)} | No-Decay (Bias/Norm) 参数张量数: {len(nodecay)}")


if __name__ == "__main__":
    demo_manual_backprop()
    demo_handwrite_adamw()
    demo_param_groups()
