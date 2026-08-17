"""Phase 2 numeric alignment: sglang_full port_opt sparse prefill kernels.

Ported verbatim from sglang_full (srt/layers/attention/port_opt/prefill.py,
the kernels the min_fake reference dumps run with under
PORT_OPT_ATTN_PREFILL=1) so the sglang-das sparse prefill attention matches
the reference bitwise.  Gated in minimax_m3.py forward_core by
PORT_CUSTOM_ATTN_SPARSE_PORTOPT=1 (only meaningful together with
PORT_CUSTOM_ATTN=1).
"""
