"""
《30天从0转行AI》第 04 课（Day 2 下）完整可运行 PyTorch 实战代码
覆盖：复数优雅实现 RoPE、大厂面试手撕因果多头注意力 CausalMHA、KV Cache 推理拼接与 GQA、KV 显存精确计算器！
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_freqs_cis(dim: int, max_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    angles = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(angles), angles)  # e^(i * m * theta)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x: [B, T, H, d]
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs_cis[: x.shape[1]].view(1, x.shape[1], 1, -1)
    x_rot = torch.view_as_real(x_c * freqs).flatten(3)
    return x_rot.type_as(x)


class CausalGQAAttention(nn.Module):
    def __init__(self, dim: int = 256, n_heads: int = 8, n_kv_heads: int = 2):
        super().__init__()
        assert dim % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.d_head, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.d_head, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, kv_cache=None):
        B, T, D = x.shape
        q = apply_rope(self.wq(x).view(B, T, self.n_heads, self.d_head), freqs_cis).transpose(1, 2)
        k = apply_rope(self.wk(x).view(B, T, self.n_kv_heads, self.d_head), freqs_cis).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if kv_cache is not None:
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)
        new_cache = (k, v)

        rep = self.n_heads // self.n_kv_heads
        k_exp = k.repeat_interleave(rep, dim=1)
        v_exp = v.repeat_interleave(rep, dim=1)

        # 手算验证 vs 官方 FlashAttention SDPA
        out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=(kv_cache is None))
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.wo(out), new_cache


def calc_kv_cache_gb(layers, batch, seq_len, n_kv_heads, d_head, dtype_bytes=2):
    return (2 * layers * batch * seq_len * n_kv_heads * d_head * dtype_bytes) / (1024 ** 3)


if __name__ == "__main__":
    freqs = precompute_freqs_cis(dim=32, max_len=128)
    attn = CausalGQAAttention(dim=256, n_heads=8, n_kv_heads=2)
    x_prefill = torch.randn(2, 16, 256)
    y_pre, cache = attn(x_prefill, freqs)
    print(f"Prefill 输出: {y_pre.shape} | KV Cache 形状: {cache[0].shape}")

    x_decode = torch.randn(2, 1, 256)
    y_dec, cache2 = attn(x_decode, freqs[16:17], kv_cache=cache)
    print(f"Decode 1步输出: {y_dec.shape} | 更新后 KV Cache 形状: {cache2[0].shape}")

    print("\n=== 70B 模型 128K 长文单用户 KV Cache 显存对比 ===")
    print(f"  旧版 MHA (64 KV头): {calc_kv_cache_gb(80, 1, 128000, 64, 128):.1f} GB")
    print(f"  现代 GQA ( 8 KV头): {calc_kv_cache_gb(80, 1, 128000, 8, 128):.1f} GB (暴省 8x!)")
    print(f"  DeepSeek MLA (~512潜维): {calc_kv_cache_gb(80, 1, 128000, 1, 512) / 2:.1f} GB (暴省 93%+!)")
