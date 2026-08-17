"""Minimal bitonic topk helpers for the ported sparse prefill kernels.

Extracted verbatim from sglang_full port_opt/sparse_decode.py (Phase 2
numeric alignment: PORT_CUSTOM_ATTN_SPARSE_PORTOPT).  The full sparse_decode
module is decode-only machinery; only the two @triton.jit helpers used by
prefill.py are copied here.
"""

import triton
import triton.language as tl

@triton.jit
def _compare_and_swap(
    x,
    ids,
    flip,
    i: tl.constexpr,
    n_dims: tl.constexpr,
):
    """Bitonic compare-and-swap (official flash_with_topk_idx.py L539-571).

    Tie-break per PLAN.md Task 4 Step 2: larger score wins; on equal scores
    the smaller index wins (for descending phases, flip != 0); for ascending
    phases the mirror rule is used (irrelevant for the final result, the
    descending final phase fixes the order).
    """
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    # idx
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    # actual compare-and-swap.  descending (flip=1): swap when the right
    # element dominates (larger score, or equal score with smaller index);
    # ascending (flip=0): mirror rule.
    swap_desc = (left < right) | ((left == right) & (left_idx > right_idx))
    swap_asc = (left > right) | ((left == right) & (left_idx < right_idx))
    cond = tl.where(flip != 0, swap_desc, swap_asc)
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x,
    ids,
    stage: tl.constexpr,
    order: tl.constexpr,
    n_dims: tl.constexpr,
):
    """Bitonic merge stage (official flash_with_topk_idx.py L574-603)."""
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in
    # ascending or descending order.
    if order == 2:
        shape: tl.constexpr = [
            n_outer * 2 ** (n_dims - 1 - stage),
            2,
            2**stage,
        ]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


