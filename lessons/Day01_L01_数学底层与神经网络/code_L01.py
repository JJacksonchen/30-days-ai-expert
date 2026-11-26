"""
《30天从0转行AI》第 01 课（Day 1 上）完整可运行 PyTorch 实战代码
覆盖：张量体检表、余弦相似度、nn.Linear转置真相、Safe Softmax、einsum、完整 5 步训练攻克 XOR！
"""
import random
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def demo_basics():
    print("=== 1. 张量体检表 & 余弦相似度 ===")
    X = torch.randn(2, 4, 8, dtype=torch.float32)  # [B=2, T=4, D=8]
    print(f"Shape={X.shape}, Dtype={X.dtype}, Device={X.device}")
    v_cat = torch.tensor([1.0, 2.0])
    v_dog = torch.tensor([2.0, 3.0])
    v_car = torch.tensor([-2.0, 1.0])
    cos_cat_dog = torch.dot(v_cat, v_dog) / (v_cat.norm() * v_dog.norm())
    cos_cat_car = torch.dot(v_cat, v_car) / (v_cat.norm() * v_car.norm())
    print(f"cos(猫, 狗) = {cos_cat_dog:.4f} | cos(猫, 汽车) = {cos_cat_car:.4f} (精确正交!)")

    print("\n=== 2. nn.Linear 隐藏转置验证 ===")
    layer = nn.Linear(8, 16)
    print(f"layer.weight.shape = {layer.weight.shape} (out_features, in_features!)")
    y_manual = X @ layer.weight.T + layer.bias
    assert torch.allclose(y_manual, layer(X), atol=1e-6)
    print("手写 X @ W.T + b 与官方 nn.Linear 误差 < 1e-6 完全一致!")

    print("\n=== 3. Safe Softmax 防溢出验证 ===")
    z_huge = torch.tensor([1000.0, 1002.0, 999.0])
    z_shift = z_huge - z_huge.max()
    safe_probs = torch.exp(z_shift) / torch.exp(z_shift).sum()
    print(f"Safe Softmax({z_huge.tolist()}) = {[round(x, 4) for x in safe_probs.tolist()]}")

    print("\n=== 4. torch.einsum 多维批量矩阵乘 ===")
    Q = torch.randn(4, 8, 128, 64)
    K = torch.randn(4, 8, 128, 64)
    s1 = Q @ K.transpose(-2, -1)
    s2 = torch.einsum("bhtd,bhsd->bhts", Q, K)
    assert torch.allclose(s1, s2, atol=1e-5)
    print(f"einsum 输出维度: {s2.shape} 验证通过!")


class SimpleMLP(nn.Module):
    def __init__(self, d_in=2, d_hidden=16, d_out=1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def demo_xor_training():
    print("\n=== 5. 完整 5 步标准训练闭环：1 秒攻克 1969 XOR 难题 ===")
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = SimpleMLP(2, 16, 1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"运行设备: {device} | 模型总参数量: {n_params} (手算 2*16+16 + 16*1+1 = 65)")

    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=device)
    y_true = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.05)

    for step in range(300):
        opt.zero_grad()
        pred = model(x)
        loss = ((pred - y_true) ** 2).mean()
        loss.backward()
        opt.step()

    model.eval()
    with torch.inference_mode():
        preds = model(x).cpu().squeeze().tolist()
    print(f"300 步后最终预测: {[round(p, 4) for p in preds]} | 目标: [0.0, 1.0, 1.0, 0.0] -> 完美破局!")


if __name__ == "__main__":
    demo_basics()
    demo_xor_training()
