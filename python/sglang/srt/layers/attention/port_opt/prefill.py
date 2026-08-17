"""Sparse prefill attention: Triton score / topk / sparse-attn kernels
(Phase 6 L3, PORT_OPT_ATTN_PREFILL).

Implementation source: triton self-written, structure following the official
``sglang_full/srt/layers/attention/minimax_sparse_ops/prefill/`` kernels:

  * score kernel    -- official ``_flash_attn_fwd_with_block_score_kernel``
    (``flash_with_topk_idx.py`` L72-284), score-only variant (this model has
    ``sparse_disable_index_value`` True for all 57 layers -> idx_o is never
    computed; official score-only path).
  * topk kernel     -- official ``_topk_index_kernel`` (L364-484), streaming
    bitonic top-K over the visible blocks.  The prefill/decode difference is
    the per-query visible mask: decode has visible == num_blocks, prefill
    has visible == pos // 128 + 1 (absolute position) and invisible blocks
    MUST be masked -inf (official ``valid_blocks``, flash_with_topk_idx.py
    L421; the decode port has no such step).
  * sparse-attn kernel -- official ``_gqa_share_sparse_fwd_kernel``
    (``topk_sparse.py`` L51-253) with the M3 layout BLOCK_SIZE_Q=1 /
    BLOCK_SIZE_H=16 (one token x the whole GQA group per program).

Numerics follow the hard constraints of ``work/phase6/L3_DESIGN.md`` §2/§3
(golden = minimal_inference ``compute_sparse_attention``, cos >= 0.9999):

  * idx scores: ``tl.dot`` with bf16 inputs and fp32 accumulation
    (``input_precision="ieee"`` -- the Task 1 MFMA smoke config validated on
    this DCU, smoke_tl_dot.py), then ``idx_scale`` applied AFTER the block
    max (spec §3.1; monotone fp32 rounding makes max(x)*c == max(x*c)
    bitwise).
  * block max: FULL block including the block's future tokens (spec §2.1,
    review F4 -- NOT the official L216 per-query clipped variant; golden
    does not clip the max).  The K loop streams to each row's
    last-visible-block end ``min(N_row, (pos//128 + 1) * 128)`` (pos =
    ABSOLUTE position); multi-row Q blocks take the max over their rows and
    mask per-row inside the loop (a row's blocks beyond its visible window
    contribute -inf and are never read by topk).
  * per-row N constraint (spec §2.1, review F3): one program handles one
    request (official grid layout (Q blocks, B_reqs x H_idx)), so a short
    request never reads blocks beyond its own history; the K boundary mask
    ``pos < N`` clips the partial last block to ``min(s+128, N_row)``.
  * topk bias (golden semantics, minimax_m3.py L1159-1167): init blocks
    REPLACED by 1e30 (init=0 for this model), the last ``local_blocks``
    visible blocks REPLACED by 1e10 (golden value); blocks >= visible
    masked -inf; k_select = min(topk, visible), -1 padding; tie-break:
    larger score wins, equal scores -> smaller index wins (the verified
    port_opt/sparse_decode.py bitonic pattern).
  * attention: fp32 online softmax with natural ``tl.exp``, ``scale``
    applied after the qk dot; p.v via ``tl.dot(p.to(bf16), v)`` with fp32
    accumulation (== golden ``weights.to(bf16) @ v``); per-position causal
    clipping with ABSOLUTE positions (spec §2.3, review F8a): visible
    tokens of the last selected block = ``min(block_end, pos+1) -
    block_start``; empty-chunk guard ``tl.where(lse > -inf, ...)``
    (official topk_sparse.py L216-221 pattern).
  * B>1 from the first version: multi-request batches use per-request
    programs (grid carries the request dim); each request's tokens use
    their own page-table row / seq_len / absolute positions (cu_seqlens
    maps tokens to requests).
"""

import torch
import triton
import triton.language as tl

from .bitonic import _bitonic_merge  # noqa: F401  (reused verbatim)

# ---------------------------------------------------------------------------
# DCU-FIX launch configs, reused verbatim from the official kernels and the
# verified decode port (port_opt/sparse_decode.py):
#
# * score kernel: official autotune keeps only num_stages=1 for
#   BLOCK_SIZE_K=128 (shared memory would exceed 65536 with ns>=2,
#   flash_with_topk_idx.py L34-37 / L56-61); BLOCK_SIZE_Q=64 x
#   BLOCK_SIZE_K=128 with num_warps=4 is the official minimal-resource
#   config; the DCU extra kargs (waves_per_eu=1, matrix_instr_nonkdim=16,
#   kpack=2) come from decode_attention.py L477-481 and were validated by
#   the Task 1 tl.dot smoke on this DCU.
# * topk: official DCU-FIX left only {BLOCK_SIZE_K: 64, num_warps=2,
#   num_stages=2} (flash_with_topk_idx.py L613-621; decode port uses the
#   same for the partial kernel).
# * sparse-attn: num_stages=1 (topk_sparse.py L23-30 DCU-FIX).
# ---------------------------------------------------------------------------
_NUM_WARPS = 4
_NUM_STAGES = 1
_DCU_FIX_KARGS = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}


@triton.jit
def _prefill_score_kernel(
    IDX_Q,             # [B, num_idx_heads, idx_head_dim] bf16 (B = extend tokens)
    IDX_K_CACHE,       # [max_kv_slots, 1, idx_head_dim] bf16 paged index K
    REQ_TO_TOKEN,      # [alloc_size, max_kv_len] int32 page table
    REQ_POOL_INDICES,  # [num_reqs] int64 page-table row per request
    SEQ_LENS,          # [num_reqs] int32 total length (prefix+extend) per request
    POSITIONS,         # [B] int32 ABSOLUTE position per extend token
    CU_SEQLENS,        # [num_reqs+1] int32 token offsets of each request in q
    BLOCK_SCORES,      # [num_idx_heads, B, max_seqblock] fp32, pre-filled -inf
    idx_scale,         # golden index scale, applied AFTER the block max
    NUM_IDX_HEADS,     # runtime: real idx head count (grid dim 1 divisor)
    head_dim,          # runtime: idx_head_dim
    MAX_KV_SLOTS,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kd,
    stride_r2tb,
    stride_pb,
    stride_cs,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    BLOCK_SIZE_Q: tl.constexpr,  # 64 (official DCU-FIX score config)
    BLOCK_SIZE_K: tl.constexpr,  # 128 == block_size (one score block per tile)
    BLOCK_SIZE_D: tl.constexpr,  # pow2(idx_head_dim)
):
    """Block-score kernel (official _flash_attn_fwd_with_block_score_kernel,
    score-only).

    Grid: (cdiv(max_q_len, BLOCK_SIZE_Q), num_reqs * num_idx_heads).  Each
    program handles one (Q block, request, idx head): the request is
    resolved from the grid's second dimension (official L132-135), so all
    rows of a Q block share the request's page-table row / seq_len and the
    per-row N constraint (F3) holds by construction; per-row positions come
    from the absolute POSITIONS array (per-row causal visible bound).

    K loop bound (F4): hi = max over rows of min(N, (pos//128+1)*128) -- the
    last-visible-block end; the FULL block max covers every block inside
    [0, hi) including the block's future tokens (golden semantics, no
    per-query clipping; blocks >= visible are never written and stay -inf,
    topk masks them out).  The only in-block clipping is the request's K
    boundary (pos < N), which is exactly golden's e = min(s+128, N).
    """
    pid_q, pid_bh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bh // NUM_IDX_HEADS
    pid_h = pid_bh % NUM_IDX_HEADS
    q_start = tl.load(CU_SEQLENS + pid_b)
    q_end = tl.load(CU_SEQLENS + pid_b + 1)
    q_len = q_end - q_start
    if pid_q * BLOCK_SIZE_Q >= q_len:
        return
    N = tl.load(SEQ_LENS + pid_b).to(tl.int32)
    req_pool_idx = tl.load(REQ_POOL_INDICES + pid_b).to(tl.int64)
    r2t_base = REQ_TO_TOKEN + req_pool_idx * stride_r2tb
    # offsets
    off_row = tl.arange(0, BLOCK_SIZE_Q)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_k = tl.arange(0, BLOCK_SIZE_K)
    dim_mask = off_d < head_dim
    row_mask = off_row < q_len - pid_q * BLOCK_SIZE_Q
    # per-row absolute positions -> per-row last-visible-block end
    row_pos = tl.load(
        POSITIONS
        + (q_start + pid_q * BLOCK_SIZE_Q + off_row) * stride_pb,
        mask=row_mask,
        other=0,
    ).to(tl.int32)
    hi_row = tl.minimum(N, ((row_pos // BLOCK_SIZE_K) + 1) * BLOCK_SIZE_K)
    hi_block = tl.max(hi_row)  # F4: loop bound = max over the block's rows
    # load idx_q [BLOCK_SIZE_Q, BLOCK_SIZE_D] bf16 for head pid_h
    q = tl.load(
        IDX_Q
        + (q_start + pid_q * BLOCK_SIZE_Q + off_row)[:, None] * stride_qb
        + pid_h * stride_qh
        + off_d[None, :] * stride_qd,
        mask=row_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    num_blocks = (N + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    for i in tl.range(0, hi_block, BLOCK_SIZE_K):
        # paged load index K via req_to_token: pos -> slot -> idx_k_cache
        pos_k = i + off_k
        pos_mask = pos_k < N
        slots = tl.load(r2t_base + pos_k, mask=pos_mask, other=0).to(tl.int64)
        # page-table bounds guard (official flash_with_topk_idx.py L203)
        slots = (slots + MAX_KV_SLOTS) % MAX_KV_SLOTS
        k = tl.load(
            IDX_K_CACHE
            + slots[None, :] * stride_kbs
            + off_d[:, None] * stride_kd,
            mask=pos_mask[None, :] & dim_mask[:, None],
            other=0.0,
        )
        # qk: bf16 x bf16 with fp32 accumulation (Task 1 smoke config)
        qk = tl.dot(q, k, input_precision="ieee")
        # per-row visible-end bound (F4): rows whose window ends before this
        # K tile are fully masked; the block max still covers the FULL block
        # (no per-query clipping inside a visible block)
        qk = tl.where((i + off_k)[None, :] < hi_row[:, None], qk, float("-inf"))
        # K boundary mask (F3): positions beyond the request's history
        # contribute -inf (partial last block -> e = min(s+128, N))
        qk = tl.where(pos_mask[None, :], qk, float("-inf"))
        # FULL block-level fp32 max (official L220-223, SCORE_TYPE="max");
        # one tile == one score block, so the tile max IS the block max
        sub_max = tl.max(qk, axis=1)
        # golden idx_scale applied AFTER the block max (spec §3.1):
        # max(x * c) == max(x) * c bitwise under monotone fp32 rounding
        sub_max = sub_max * idx_scale
        block_idx = i // BLOCK_SIZE_K
        tl.store(
            BLOCK_SCORES
            + pid_h * stride_s_h
            + (q_start + pid_q * BLOCK_SIZE_Q + off_row) * stride_s_n
            + block_idx * stride_s_k,
            sub_max,
            mask=row_mask & (block_idx < num_blocks),
        )


@triton.jit
def _prefill_topk_kernel(
    s_ptr,            # [num_idx_heads, B, max_seqblock] fp32 block scores
    ti_ptr,           # topk_idx: [num_idx_heads, B, topk] int32 (-1 padded)
    POSITIONS,        # [B] int32 ABSOLUTE position per extend token
    CU_SEQLENS,       # [num_reqs+1] int32 token offsets of each request in q
    block_size: tl.constexpr,
    topk: tl.constexpr,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    NUM_IDX_HEADS,    # runtime
    stride_s_h, stride_s_n, stride_s_k,
    stride_ti_h, stride_ti_n, stride_ti_t,
    BLOCK_SIZE_K: tl.constexpr,   # 64 (official DCU-FIX topk config)
    BLOCK_SIZE_T: tl.constexpr,   # pow2(topk)
):
    """Per-query bitonic top-K over the visible blocks (official
    _topk_index_kernel, flash_with_topk_idx.py L364-484).

    Grid: (max_q_len, num_reqs * num_idx_heads) -- one (token, idx head)
    per program (official block_size_q=1 layout).  The prefill/decode
    difference: visible = pos // block_size + 1 (absolute position), and
    blocks >= visible are masked -inf (official valid_blocks L421; the
    decode port needs no such step).  init/local bias follows golden
    REPLACEMENT semantics (minimax_m3.py L1159-1167): init blocks -> 1e30,
    the last ``local_blocks`` visible blocks -> 1e10; k_select =
    min(topk, visible), the rest -1 padded.
    """
    pid_q, pid_bh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bh // NUM_IDX_HEADS
    pid_h = pid_bh % NUM_IDX_HEADS
    q_start = tl.load(CU_SEQLENS + pid_b)
    q_end = tl.load(CU_SEQLENS + pid_b + 1)
    q_len = q_end - q_start
    if pid_q >= q_len:
        return
    token = q_start + pid_q
    pos = tl.load(POSITIONS + token).to(tl.int32)
    visible = (pos // block_size) + 1
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    s_ptrs = (
        s_ptr + pid_h * stride_s_h + token * stride_s_n + off_k * stride_s_k
    )
    # streaming top-K state (official L417-419)
    topk_score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2
    for i in tl.range(0, visible, BLOCK_SIZE_K):
        # masks (official L422-426)
        causal_mask = i + off_k < visible
        init_mask = i + off_k < init_blocks
        local_mask = (i + off_k >= visible - local_blocks) & (
            i + off_k < visible
        )
        # load score; invisible blocks -> -1e30 sentinel (per-query visible
        # mask, the prefill/decode difference)
        score = tl.load(s_ptrs, mask=causal_mask, other=-1e30).to(tl.float32)
        # handle NaN: NaN inputs cause bitonic sort to fail, resulting in
        # invalid indices (-2) in the topk list (official L431)
        score = tl.where(score != score, -1e30, score)
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        # init/local bias, golden REPLACEMENT semantics (init=0 for this
        # model; official L436-443 with golden values 1e30/1e10)
        score = tl.where(causal_mask & init_mask, 1e30, score)
        score = tl.where(causal_mask & local_mask, 1e10, score)
        # bitonic merge (verified decode-port pattern)
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = (
            tl.where(causal_mask, i + off_k + 1, 0),  # 1-indexed global
            topk_idx,
        )
        n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), j, 2, n_dims
            )
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims
            )
            topk_score_new = last_topk_score * left_half_mask + topk_score * (
                1 - left_half_mask
            )
            topk_idx_new = last_topk_idx * left_half_mask + topk_idx * (
                1 - left_half_mask
            )
            topk_score, topk_idx = _bitonic_merge(
                topk_score_new, topk_idx_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims
            )
    # extract first BLOCK_SIZE_T entries (top-K, descending)
    topk_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    final_idx = tl.sum(
        topk_mask[:, None]
        * tl.reshape(topk_idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )
    # golden k_select = min(topk, visible); the rest -1 padded (official
    # L477-484)
    ti_ptrs = (
        ti_ptr + pid_h * stride_ti_h + token * stride_ti_n + off_t * stride_ti_t
    )
    final_idx = tl.where(off_t < tl.minimum(topk, visible), final_idx, -1)
    tl.store(ti_ptrs, final_idx.to(ti_ptrs.dtype.element_ty))


@triton.jit
def _prefill_sparse_attn_kernel(
    Q,               # [B, num_q_heads, head_dim] bf16 (B = extend tokens)
    K_CACHE,         # [max_kv_slots, num_kv_heads, head_dim] bf16 paged
    V_CACHE,         # [max_kv_slots, num_kv_heads, head_dim] bf16 paged
    TOPK_IDX,        # [num_idx_heads, B, topk] int32 (0-indexed, -1=invalid)
    O,               # [B, num_q_heads, head_dim] bf16 output
    REQ_TO_TOKEN,    # [alloc_size, max_kv_len] int32 page table
    REQ_POOL_INDICES,  # [num_reqs] int64 page-table row per request
    SEQ_LENS,        # [num_reqs] int32 total length per request
    POSITIONS,       # [B] int32 ABSOLUTE position per extend token
    CU_SEQLENS,      # [num_reqs+1] int32 token offsets of each request in q
    MAX_KV_SLOTS,
    gqa_group_size,  # runtime num_q_heads // num_kv_heads
    head_dim,        # runtime
    max_topk,        # runtime: topk_idx.shape[2]
    scale,           # applied after the qk dot
    stride_qb, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_r2tb,
    stride_ti_h, stride_ti_n, stride_ti_t,
    stride_ob, stride_oh, stride_od,
    BLOCK_SIZE_H: tl.constexpr,   # pow2(gqa_group_size) (M3: 16)
    BLOCK_SIZE_N: tl.constexpr,   # block_size (128)
    BLOCK_SIZE_D: tl.constexpr,   # pow2(head_dim)
    BLOCK_SIZE_T: tl.constexpr,   # pow2(max_topk)
):
    """Sparse attention over the SELECTED blocks (official
    _gqa_share_sparse_fwd_kernel, topk_sparse.py L51-253) with the M3
    layout BLOCK_SIZE_Q=1 / BLOCK_SIZE_H=16 (one token x the whole GQA
    group per program).

    Grid: (max_q_len, num_kv_heads, num_reqs).  Only the topk-selected
    blocks are gathered (paged K/V via req_to_token) and accumulated with
    the fp32 online softmax (natural exp, scale after the qk dot; p.v via
    tl.dot(p.to(bf16), v) fp32 acc == golden weights.to(bf16) @ v).

    Per-position causal clipping with ABSOLUTE positions (spec §2.3, F8a):
    the visible tokens of a selected block are
    [c, min(c+128, pos+1)) -- for every non-last selected block this is the
    full block, for the last visible block it clips the query's own future
    tokens inside the block.  (decode relies on cache completeness and has
    no such step; prefill MUST clip by absolute position -- relative
    positions would mis-clip for chunk >= 2.)
    """
    pid_q, pid_kh, pid_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    q_start = tl.load(CU_SEQLENS + pid_b)
    q_end = tl.load(CU_SEQLENS + pid_b + 1)
    q_len = q_end - q_start
    if pid_q >= q_len:
        return
    token = q_start + pid_q
    N = tl.load(SEQ_LENS + pid_b).to(tl.int32)
    pos = tl.load(POSITIONS + token).to(tl.int32)  # absolute position
    req_pool_idx = tl.load(REQ_POOL_INDICES + pid_b).to(tl.int64)
    r2t_base = REQ_TO_TOKEN + req_pool_idx * stride_r2tb
    pid_h = pid_kh * gqa_group_size
    # offsets
    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    dim_mask = off_d < head_dim
    h_mask = off_h < gqa_group_size
    # load q [BLOCK_SIZE_H, BLOCK_SIZE_D] bf16 (rows >= gqa_group_size are
    # masked to zero; never stored)
    q = tl.load(
        Q
        + token * stride_qb
        + (pid_h + off_h)[:, None] * stride_qh
        + off_d[None, :] * stride_qd,
        mask=h_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    # load the selected block indices (right-padded with -1)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    idx_base = TOPK_IDX + pid_kh * stride_ti_h + token * stride_ti_n
    topk_idx = tl.load(
        idx_base + off_t * stride_ti_t, mask=off_t < max_topk, other=-1
    )
    valid_idx = tl.where(topk_idx >= 0, off_t, -1)
    real_topk = tl.sum(valid_idx != -1, axis=0)
    # statistics kept at -inf so an empty selection falls out cleanly
    # (official L216-221 guard)
    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    cur_idx_ptr = idx_base
    # sparse loop: only the selected blocks are touched (official L188-240)
    for _ in tl.range(0, real_topk):
        # block start (absolute K position)
        c = tl.load(cur_idx_ptr).to(tl.int32) * BLOCK_SIZE_N
        cur_idx_ptr = cur_idx_ptr + stride_ti_t
        # per-position causal clip (absolute position): visible tokens of
        # this block = min(c+128, pos+1) - c
        n_vis = tl.minimum(c + BLOCK_SIZE_N, pos + 1) - c
        # paged load K/V via req_to_token: pos -> slot -> cache
        pos_k = c + off_n
        pos_mask = pos_k < N
        slots = tl.load(
            r2t_base + pos_k, mask=pos_mask, other=0,
        ).to(tl.int64)
        # page-table bounds guard (official flash_with_topk_idx.py L203)
        slots = (slots + MAX_KV_SLOTS) % MAX_KV_SLOTS
        # k shape: [BLOCK_SIZE_D, BLOCK_SIZE_N] (transposed for tl.dot)
        k = tl.load(
            K_CACHE
            + slots[None, :] * stride_kbs
            + pid_kh * stride_kh
            + off_d[:, None] * stride_kd,
            mask=pos_mask[None, :] & dim_mask[:, None],
            other=0.0,
        )
        # v shape: [BLOCK_SIZE_N, BLOCK_SIZE_D]
        v = tl.load(
            V_CACHE
            + slots[:, None] * stride_vbs
            + pid_kh * stride_vh
            + off_d[None, :] * stride_vd,
            mask=pos_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        # qk: bf16 x bf16 fp32 accumulate, scale AFTER the dot (golden +
        # official topk_sparse.py L217); causal clip and the N boundary
        # mask positions beyond the visible window to -inf
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_N), dtype=tl.float32)
        qk += tl.where(off_n[None, :] < n_vis, 0, float("-inf"))
        qk += tl.where(off_n[None, :] < N - c, 0, float("-inf"))
        qk += tl.dot(q, k, input_precision="ieee") * scale
        # online softmax in fp32 with natural exp (official L221-240)
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o_scale = tl.exp(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        # p.v: tl.dot(p.to(bf16), v) fp32 accumulate == golden
        # weights.to(bf16) @ v (masked-out positions: exp(-inf) == 0)
        acc_o += tl.dot(p.to(v.dtype), v)
        # update statistics
        m_i = m_ij
        lse_i = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)
    # final scale.  Empty selections keep m_i = lse_i = -inf and the naive
    # exp(-inf - (-inf)) = NaN would poison the output; the official
    # L216-221 guard emits clean zeros.
    scale_final = tl.where(
        lse_i > float("-inf"),
        tl.exp(m_i - lse_i),
        tl.zeros_like(lse_i),
    )
    acc_o = acc_o * scale_final[:, None]
    tl.store(
        O
        + token * stride_ob
        + (pid_h + off_h)[:, None] * stride_oh
        + off_d[None, :] * stride_od,
        acc_o.to(O.dtype.element_ty),
        mask=h_mask[:, None] & dim_mask[None, :],
    )


def prefill_sparse_attention(
    q,
    idx_q,
    k_cache,
    v_cache,
    idx_k_cache,
    req_to_token,
    req_pool_indices,
    seq_lens,
    positions,
    scale,
    idx_scale,
    block_size,
    topk_blocks,
    local_blocks,
    num_q_heads,
    num_kv_heads,
    num_idx_heads,
    head_dim,
    idx_head_dim,
    max_kv_slots,
    cu_seqlens=None,
    return_topk_idx=False,
) -> torch.Tensor:
    """Triton sparse block-topk PREFILL attention (L3_DESIGN.md §2.4).

    Three-kernel pipeline (official prefill structure):

      1. score kernel  -> block_scores [num_idx_heads, B, max_seqblock]
      2. topk kernel   -> topk_idx [num_idx_heads, B, topk_blocks]
      3. sparse-attn   -> o [B, num_q_heads, head_dim] bf16

    Args:
        q: [B, num_q_heads, head_dim] bf16 queries, B = total extend tokens
            concatenated per request (sglang prefill layout).
        idx_q: [B, num_idx_heads, idx_head_dim] bf16 index-head queries.
        k_cache / v_cache: [max_kv_slots, num_kv_heads, head_dim] bf16
            paged KV buffers (TokenToKVPool layout), already containing the
            current extend tokens (the caller writes them first).
        idx_k_cache: [max_kv_slots, 1, idx_head_dim] bf16 paged index K.
        req_to_token: [alloc_size, max_kv_len] int32 page table; the row
            ``req_pool_indices[b]`` maps request b's token positions to KV
            slots (the request's full history, prefix + extend).
        req_pool_indices: [num_reqs] int64 page-table row per request.
        seq_lens: [num_reqs] per-request TOTAL sequence lengths (prefix +
            extend; the K-side history length), GPU tensor or list.
        positions: [B] per-token ABSOLUTE positions (drives the causal
            visible-block boundary and the in-block causal clip).
        scale: QK scale, applied after the fp32 qk dot.
        idx_scale: index scale, applied after the block max (golden).
        block_size: sparse block size (128); topk_blocks: selected blocks
            (16); local_blocks: last visible blocks biased (1).
        num_q_heads / num_kv_heads / num_idx_heads / head_dim /
            idx_head_dim / max_kv_slots: shape parameters.
        cu_seqlens: [num_reqs+1] int32 token offsets of each request within
            the concatenated q (required for B>1; defaults to the B=1 case
            [0, B] when None).
        return_topk_idx: when True, also return the selected block index
            tensor [num_idx_heads, B, topk_blocks] int32 (0-indexed,
            -1=invalid) as a second value.

    Returns:
        [B, num_q_heads, head_dim] bf16 attention output (or a
        (output, topk_idx) tuple when return_topk_idx is True).
    """
    B = q.shape[0]
    assert B > 0, "empty prefill batch"
    assert q.shape[1] == num_q_heads and q.shape[2] == head_dim
    assert idx_q.shape[1] == num_idx_heads and idx_q.shape[2] == idx_head_dim
    assert num_idx_heads == num_kv_heads, (
        "this model shards the idx heads exactly like the kv heads per rank, "
        "so the topk_idx tensor is shared between the topk and sparse-attn "
        "stages without topk_index_reduce"
    )
    assert block_size & (block_size - 1) == 0, "block_size must be a power of 2"
    assert positions.shape[0] == B

    if cu_seqlens is None:
        assert req_pool_indices.shape[0] == 1, (
            "cu_seqlens is required for B>1 prefill batches"
        )
        cu_seqlens_gpu = torch.tensor([0, B], dtype=torch.int32, device=q.device)
    else:
        cu_seqlens_gpu = cu_seqlens.to(device=q.device, dtype=torch.int32)
    num_reqs = cu_seqlens_gpu.shape[0] - 1
    assert num_reqs == req_pool_indices.shape[0]
    # per-request q lengths (one sync; prefill is eager, not cuda-graph)
    q_lens = cu_seqlens_gpu[1:] - cu_seqlens_gpu[:-1]
    max_q_len = int(q_lens.max().item())
    # L3-2 D2 hardening: a broken cu_seqlens (e.g. duplicated leading zero
    # from a caller bug) makes every kernel program take the empty-batch
    # early return -- the outputs would silently stay torch.empty garbage.
    # Fail loudly instead (server path always satisfies sum(q_lens) == B).
    assert int(q_lens.sum().item()) == B, (
        f"cu_seqlens inconsistent with q batch: sum(q_lens)="
        f"{int(q_lens.sum().item())} != B={B} (cu_seqlens={cu_seqlens_gpu.tolist()})"
    )

    # per-request total lengths (K side); int32 GPU tensor (the decode port
    # conversion: capture-safe dtype conversion in eager)
    if isinstance(seq_lens, (list, tuple)):
        seq_lens_gpu = torch.tensor(seq_lens, dtype=torch.int32, device=q.device)
    else:
        seq_lens_gpu = seq_lens.to(device=q.device, dtype=torch.int32)
    # absolute positions -> int32 (HIP scheduler tensor is int64)
    positions_gpu = positions.to(device=q.device, dtype=torch.int32)

    max_kv_len = req_to_token.shape[1]
    max_seqblock = (max_kv_len + block_size - 1) // block_size
    gqa_group_size = num_q_heads // num_kv_heads
    # official topk_sparse.py L290 constraint with block_size_q=1 (the M3
    # layout: one token x the whole GQA group per attn program)
    assert gqa_group_size * 1 <= 128
    block_size_t = triton.next_power_of_2(topk_blocks)

    # ---- Stage 1: block scores -------------------------------------------
    block_scores = torch.full(
        (num_idx_heads, B, max_seqblock),
        fill_value=-float("inf"),
        dtype=torch.float32,
        device=q.device,
    )
    grid = (triton.cdiv(max_q_len, 64), num_reqs * num_idx_heads)
    _prefill_score_kernel[grid](
        idx_q,
        idx_k_cache,
        req_to_token,
        req_pool_indices,
        seq_lens_gpu,
        positions_gpu,
        cu_seqlens_gpu,
        block_scores,
        idx_scale,
        num_idx_heads,
        idx_head_dim,
        max_kv_slots,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        idx_k_cache.stride(0),
        idx_k_cache.stride(2),
        req_to_token.stride(0),
        positions_gpu.stride(0),
        cu_seqlens_gpu.stride(0),
        block_scores.stride(0),
        block_scores.stride(1),
        block_scores.stride(2),
        BLOCK_SIZE_Q=64,
        BLOCK_SIZE_K=block_size,
        BLOCK_SIZE_D=triton.next_power_of_2(idx_head_dim),
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
        **_DCU_FIX_KARGS,
    )

    # ---- Stage 2: per-query top-k over the visible blocks ----------------
    topk_idx = torch.empty(
        (num_idx_heads, B, topk_blocks),
        dtype=torch.int32,
        device=q.device,
    )
    grid = (max_q_len, num_reqs * num_idx_heads)
    _prefill_topk_kernel[grid](
        block_scores,
        topk_idx,
        positions_gpu,
        cu_seqlens_gpu,
        block_size,
        topk_blocks,
        0,  # init_blocks: 0 for this model (no init bias)
        local_blocks,
        num_idx_heads,
        block_scores.stride(0),
        block_scores.stride(1),
        block_scores.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        BLOCK_SIZE_K=64,  # official DCU-FIX: only {64, nw=2, ns=2} safe
        BLOCK_SIZE_T=block_size_t,
        num_warps=2,
        num_stages=2,
    )

    # ---- Stage 3: sparse attention over the selected blocks --------------
    out = torch.empty(
        (B, num_q_heads, head_dim), dtype=q.dtype, device=q.device
    )
    grid = (max_q_len, num_kv_heads, num_reqs)
    _prefill_sparse_attn_kernel[grid](
        q,
        k_cache,
        v_cache,
        topk_idx,
        out,
        req_to_token,
        req_pool_indices,
        seq_lens_gpu,
        positions_gpu,
        cu_seqlens_gpu,
        max_kv_slots,
        gqa_group_size,
        head_dim,
        topk_blocks,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        req_to_token.stride(0),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_SIZE_H=max(16, triton.next_power_of_2(gqa_group_size)),
        BLOCK_SIZE_N=block_size,
        BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
        BLOCK_SIZE_T=block_size_t,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
        **_DCU_FIX_KARGS,
    )
    if return_topk_idx:
        return out, topk_idx
    return out
