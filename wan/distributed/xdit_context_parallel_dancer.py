# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
torch.backends.cudnn.deterministic = True
# import torch.cuda.amp as amp
import torch.amp as amp
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention

from ..modules.model_dancer import sinusoidal_embedding_1d

from einops import rearrange


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


# @amp.autocast(enabled=False)
@amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16)
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    # return torch.stack(output).float()
    return torch.stack(output)


def usp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    condition=None,
    ref_x=None,
    ref_c=None,
    clip_fea_x=None,
    clip_fea_c=None,
    clip_fea=None,
    y=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert clip_fea_x is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    x_noise_clone = torch.stack(x)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # Temporal Motion Coherence Module.
    condition_temporal = [self.condition_embedding_temporal(c.unsqueeze(0)) for c in [condition]]
    
    # Spatial Structure Adaptive Extractor.
    with amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
        condition = condition[None]
        bs, _, time_steps, _, _ = condition.shape
        condition_reshape = rearrange(condition, 'b c t h w -> (b t) c h w')
        condition_spatial = self.condition_embedding_spatial(condition_reshape)
        condition_spatial = rearrange(condition_spatial, '(b t) c h w -> b c t h w', t=time_steps, b=bs)

    # Hierarchical Aggregation (1): condition, temporal condition, spatial condition
    condition_fused = condition + condition_temporal[0] + condition_spatial

    # Frame-wise Attention Alignment Unit.
    with amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
        condition_aligned = self.condition_embedding_align(condition_fused, x_noise_clone)

    real_seq = x[0].shape[1]
    
    # Condition Fusion/Injection, Hierarchical Aggregation (2): x, fused condition, aligned condition
    x = [self.patch_embedding_fuse(torch.cat([u[None], c[None], a[None]], 1)) for u, c, a in
            zip(x, condition_fused, condition_aligned)]
    
    # Condition Augmentation: x_cond, ref_x, ref_c
    ref_x = [ref_x]
    ref_c = [ref_c]
    ref_x = [self.patch_embedding(r.unsqueeze(0)) for r in ref_x]
    ref_c = [self.patch_embedding_ref_c(r[:16].unsqueeze(0)) for r in ref_c]
    x = [torch.cat([r, u, v], dim=2) for r, u, v in zip(x, ref_x, ref_c)]

    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    seq_len = seq_lens.max()
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                    dim=1) for u in x
    ])

    # time embeddings
    with amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(x.dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea_x is not None:
        context_clip_x = self.img_emb(clip_fea_x)  # bs x 257 x dim
    if clip_fea_c is not None:
        context_clip_c = self.img_emb(clip_fea_c)  # bs x 257 x dim
    if clip_fea_x is not None:
        context_clip = context_clip_x if context_clip_c is None else context_clip_x + context_clip_c    # Condition Augmentation
        context = torch.concat([context_clip, context], dim=1)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)

    # Context Parallel
    x = torch.chunk(
        x, get_sequence_parallel_world_size(),
        dim=1)[get_sequence_parallel_rank()]

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = get_sp_group().all_gather(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    # return [u.float() for u in x]
    return [u[:, :real_seq, ...] for u in x]


def usp_attn_forward(self,
                     x,
                     seq_lens,
                     grid_sizes,
                     freqs,
                     dtype=torch.bfloat16):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    # TODO: We should use unpaded q,k,v for attention.
    # k_lens = seq_lens // get_sequence_parallel_world_size()
    # if k_lens is not None:
    #     q = torch.cat([u[:l] for u, l in zip(q, k_lens)]).unsqueeze(0)
    #     k = torch.cat([u[:l] for u, l in zip(k, k_lens)]).unsqueeze(0)
    #     v = torch.cat([u[:l] for u, l in zip(v, k_lens)]).unsqueeze(0)

    x = xFuserLongContextAttention()(
        None,
        query=half(q),
        key=half(k),
        value=half(v),
        window_size=self.window_size)

    # TODO: padding after attention.
    # x = torch.cat([x, x.new_zeros(b, s - x.size(1), n, d)], dim=1)

    # output
    x = x.to(torch.bfloat16)
    x = x.flatten(2)
    x = self.o(x)
    return x
