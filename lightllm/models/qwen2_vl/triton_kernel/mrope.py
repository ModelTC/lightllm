import time
import torch
import triton
import triton.language as tl


@triton.jit
def mrope_kernel_combined(
    Q_ptr,
    K_ptr,
    COS_ptr,
    SIN_ptr,
    AXIS_MAP_ptr,
    Q_out_ptr,
    K_out_ptr,
    B: tl.int32,
    H_q: tl.int32,
    H_k: tl.int32,
    L: tl.int32,
    D: tl.int32,
    HALF: tl.int32,
    BLOCK_D: tl.constexpr,
):
    total_h = H_q + H_k

    pid_bh = tl.program_id(0)
    pid_l = tl.program_id(1)

    b = pid_bh // total_h
    head_local = pid_bh - b * total_h

    # decide whether this head comes from q or k
    is_q = head_local < H_q
    head_q = head_local
    head_k = head_local - H_q

    base_ptr = tl.where(is_q, Q_ptr, K_ptr)
    out_ptr = tl.where(is_q, Q_out_ptr, K_out_ptr)
    h_sub = tl.where(is_q, head_q, head_k)
    H_sub = tl.where(is_q, H_q, H_k)

    # base offset for (b, h_sub, pid_l)
    base = ((b * H_sub + h_sub) * L + pid_l) * D

    offs = tl.arange(0, BLOCK_D)
    idx = base + offs
    mask = offs < D

    vals = tl.load(base_ptr + idx, mask=mask, other=0.0)
    axis_id = tl.load(AXIS_MAP_ptr + offs, mask=mask, other=0)
    axis_id_b = b * 3 + axis_id

    seq_off = pid_l * D
    cos_idx = axis_id_b * (L * D) + seq_off + offs
    c = tl.load(COS_ptr + cos_idx, mask=mask, other=0.0)
    s = tl.load(SIN_ptr + cos_idx, mask=mask, other=0.0)

    # rotate_half
    rot_idx = tl.where(offs < HALF, idx + HALF, idx - HALF)
    rot_vals = tl.load(base_ptr + rot_idx, mask=mask, other=0.0)
    sign = tl.where(offs < HALF, -1.0, 1.0)
    rot_vals *= sign

    out_vals = vals * c + rot_vals * s
    tl.store(out_ptr + idx, out_vals, mask=mask)


def mrope_triton(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mrope_section):
    B, H_q, L, D = q.shape
    H_k = k.shape[1]

    # build axis_map 0/1/2 label per feature dim
    axis_map = []
    for i, n in enumerate(mrope_section * 2):
        axis_map += [i % 3] * n
    axis_map = torch.tensor(axis_map, dtype=torch.int32, device=q.device)

    cos_flat = cos.transpose(0, 1).expand(B, 3, L, D).contiguous().reshape(-1)
    sin_flat = sin.transpose(0, 1).expand(B, 3, L, D).contiguous().reshape(-1)

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    grid = (B * (H_q + H_k), L)
    mrope_kernel_combined[grid](
        q,
        k,
        cos_flat,
        sin_flat,
        axis_map,
        q_out,
        k_out,
        B,
        H_q,
        H_k,
        L,
        D,
        D // 2,
        BLOCK_D=128,
    )
    return q_out, k_out


# ----------------  test ---------------- #
def test():

    # torch实现
    def rotate_half(x: torch.Tensor):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
        chunks = mrope_section * 2
        cos_embed = torch.cat(
            [m[i % 3] for i, m in enumerate(cos.split(chunks, dim=-1))],
            dim=-1,
        ).unsqueeze(unsqueeze_dim)
        sin_embed = torch.cat(
            [m[i % 3] for i, m in enumerate(sin.split(chunks, dim=-1))],
            dim=-1,
        ).unsqueeze(unsqueeze_dim)

        q_out = q * cos_embed + rotate_half(q) * sin_embed
        k_out = k * cos_embed + rotate_half(k) * sin_embed
        return q_out, k_out

    B, H_q, H_k, L, D = 1, 28, 4, 16384, 128
    mrope_section = [16, 24, 24]
    torch.manual_seed(0)
    device = "cuda"

    q = torch.rand(B, H_q, L, D, dtype=torch.float32, device=device)
    k = torch.rand(B, H_k, L, D, dtype=torch.float32, device=device)
    cos = torch.rand(3, 1, L, D, dtype=torch.float32, device=device)
    sin = torch.rand(3, 1, L, D, dtype=torch.float32, device=device)

    # 精度对比
    ref_q, ref_k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1)

    torch.cuda.synchronize()
    out_q, out_k = mrope_triton(q, k, cos, sin, mrope_section)
    torch.cuda.synchronize()

    err_q = (out_q - ref_q).abs().max().item()
    err_k = (out_k - ref_k).abs().max().item()
    print(f"abs‑max error   q:{err_q:.6f}, k:{err_k:.6f}")

    assert err_q < 1e-2 and err_k < 1e-2

    # 速度对比
    n_iter = 100
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)

    e0.record()
    for _ in range(n_iter):
        _ = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1)
    e1.record()
    torch.cuda.synchronize()
    t_ref = e0.elapsed_time(e1) / n_iter

    e0.record()
    for _ in range(n_iter):
        _ = mrope_triton(q, k, cos, sin, mrope_section)
    e1.record()
    torch.cuda.synchronize()
    t_tri = e0.elapsed_time(e1) / n_iter

    print(f"torch {t_ref:.2f} ms/iter")
    print(f"triton {t_tri:.2f} ms/iter")
