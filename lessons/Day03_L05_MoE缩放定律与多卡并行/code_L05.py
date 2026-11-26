"""
《30天从0转行AI》第 05 课（Day 3 上）完整可运行 PyTorch 实战代码
覆盖：18行手撕 DeepSeek 风格 Sparse MoE（共享常驻专家 + Top-K稀疏路由 + 无辅助损失动态偏置均衡）+ 6ND 算力估算器！
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class DeepSeekMoELayer(nn.Module):
    def __init__(self, dim: int = 128, n_experts: int = 8, top_k: int = 2, bias_gamma: float = 0.001):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.bias_gamma = bias_gamma
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.register_buffer("router_bias", torch.zeros(n_experts))  # 无辅助损失动态偏置!
        self.shared = SwiGLU(dim, dim * 2)                           # 常驻共享专家
        self.experts = nn.ModuleList([SwiGLU(dim, dim) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        raw_scores = self.gate(x).sigmoid()                          # 原始纯净亲和分 s_i
        # 1. 选人时加上动态排队偏置 (不污染反向梯度!)
        select_scores = raw_scores + self.router_bias.view(1, 1, -1)
        _, top_idx = torch.topk(select_scores, self.top_k, dim=-1)   # [B, T, K]
        # 2. 真正干活权重仍只取纯净 raw_scores 并重归一化
        top_w = raw_scores.gather(-1, top_idx)
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-6)

        out = self.shared(x)                                         # 常驻共享专家保底
        flat_x = x.view(-1, x.size(-1))
        flat_out = out.view(-1, x.size(-1))
        flat_idx = top_idx.view(-1, self.top_k)
        flat_w = top_w.view(-1, self.top_k)

        loads = torch.zeros(self.n_experts, device=x.device)
        for e_id, expert in enumerate(self.experts):
            tok_ids, k_pos = torch.where(flat_idx == e_id)
            loads[e_id] = tok_ids.numel()
            if tok_ids.numel() > 0:
                w = flat_w[tok_ids, k_pos].unsqueeze(-1)
                flat_out[tok_ids] = flat_out[tok_ids] + w * expert(flat_x[tok_ids])

        # 3. 训练期在线动态更新负载均衡偏置 (零 Aux Loss 副作用!)
        if self.training:
            with torch.no_grad():
                avg_load = loads.mean()
                self.router_bias.add_(self.bias_gamma * torch.sign(avg_load - loads))

        return out


def estimate_pretrain_days(params_b, tokens_t, num_gpus, gpu_tflops=989.0, mfu=0.40):
    total_flops = 6.0 * (params_b * 1e9) * (tokens_t * 1e12)
    cluster_flops_s = num_gpus * (gpu_tflops * 1e12) * mfu
    return total_flops / cluster_flops_s / 86400.0


if __name__ == "__main__":
    torch.manual_seed(42)
    moe = DeepSeekMoELayer(dim=128, n_experts=8, top_k=2)
    x = torch.randn(4, 32, 128)
    for _ in range(10):
        y = moe(x)
    print(f"DeepSeekMoE 输出 Shape: {y.shape}")
    print(f"10 步动态均衡后 router_bias: {[round(v, 4) for v in moe.router_bias.tolist()]}")

    print("\n=== 6ND 大厂预训练集群耗时速算器 (H100 @ 40% MFU) ===")
    print(f"  7B 模型 + 1T Token + 64×H100    -> 约 {estimate_pretrain_days(7, 1.0, 64):.1f} 天")
    print(f"  70B 稠密 + 15T Token + 2048×H100 -> 约 {estimate_pretrain_days(70, 15.0, 2048):.1f} 天")
    print(f"  DeepSeek 37B激活 + 14.8T + 2048×H800 -> 约 {estimate_pretrain_days(37, 14.8, 2048):.1f} 天 (MoE暴省算力!)")
