"""Paged KV for the block-scaled (NVFP4 / MXFP8) SM100 forward kernel.

Layout follows the sglang / flashinfer convention:
  k, v         : (num_pages, page_size, num_kv_heads, head_dim)   [NHD]
  page_table   : (batch, max_num_pages_per_seq), int32
  mSFK, mSFV   : scale factors paged the same way (num_pages is the batch dim)

Quantization is per-token (NVFP4/MXFP8 scale-factor groups lie inside head_dim
for K, and inside a page for the K-major V), so a paged run must be *bitwise*
identical to the equivalent non-paged run: same values, same K-block order.
"""
import math
import pytest
import torch

from flash_attn.cute.interface import _flash_attn_fwd

pytest.importorskip("flashinfer.quantization")

SF_VEC, TILE_M, PAGE_SIZE = 16, 128, 128
MODES = ["nvfp4_bf16", "nvfp4_fp8", "nvfp4_fp4", "nvfp4_mxfp8", "mxfp8_bf16", "mxfp8_fp8"]


def _sm100():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 10


pytestmark = pytest.mark.skipif(not _sm100(), reason="requires SM100 (Blackwell) GPU")


def _quant_tokenmajor(ref, nvfp4: bool):
    """(b, s, h, d) -> quantized data + swizzled SF, groups along head_dim."""
    from flashinfer.quantization import nvfp4_quantize, mxfp8_quantize, SfLayout
    b, s, h, d = ref.shape
    vec = SF_VEC if nvfp4 else 32
    t2d = ref.to(torch.bfloat16).reshape(b * s, h * d)
    if nvfp4:
        one = torch.ones(1, device=ref.device, dtype=torch.float32)
        data, sf = nvfp4_quantize(t2d, one, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        data = data.reshape(b, s, h, d // 2).view(torch.uint8).view(torch.float4_e2m1fn_x2)
    else:
        data, sf = mxfp8_quantize(t2d, sf_swizzle_layout=SfLayout.layout_128x4)
        if data.dtype == torch.uint8:
            data = data.view(torch.float8_e4m3fn)
        data = data.reshape(b, s, h, d)
    rest_k = (d // vec) // 4
    sf = sf.reshape(b * (s // TILE_M), (h * (d // vec)) // 4, 32, 4, 4)
    sf = sf.reshape(b, s // TILE_M, h, rest_k, 32, 4, 4)
    sf = sf.permute(0, 2, 1, 3, 4, 5, 6).contiguous().permute(4, 5, 2, 6, 3, 1, 0)
    return data, sf


def _quant_kmajor(ref, mode):
    """(b, s, h, d) -> K-major (seqlen-contiguous) V + SF, groups along seqlen."""
    from flashinfer.quantization import nvfp4_quantize, mxfp8_quantize, SfLayout
    b, s, h, d = ref.shape
    km = ref.to(torch.bfloat16).permute(0, 2, 3, 1).contiguous().reshape(b * h * d, s)
    if mode == "fp4":
        vec = SF_VEC
        one = torch.ones(1, device=ref.device, dtype=torch.float32)
        data, sf = nvfp4_quantize(km, one, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        v = data.reshape(b, h, d, s // 2).view(torch.uint8).view(torch.float4_e2m1fn_x2)
    else:
        vec = 32
        data, sf = mxfp8_quantize(km, sf_swizzle_layout=SfLayout.layout_128x4)
        if data.dtype == torch.uint8:
            data = data.view(torch.float8_e4m3fn)
        v = data.reshape(b, h, d, s).permute(0, 3, 1, 2)
    sf = sf.reshape(b * h * (d // TILE_M), s // (vec * 4), 32, 4, 4)
    sf = sf.reshape(b, h, d // TILE_M, s // (vec * 4), 32, 4, 4)
    sf = sf.contiguous().permute(4, 5, 2, 6, 3, 1, 0)
    return v, sf


def _build(mode, q_ref, k_ref, v_ref):
    qk, pv = mode.split("_")
    nvfp4 = qk == "nvfp4"
    q, sfq = _quant_tokenmajor(q_ref, nvfp4)
    k, sfk = _quant_tokenmajor(k_ref, nvfp4)
    sfv = None
    if pv == "bf16":
        v = v_ref.to(torch.bfloat16)
    elif pv == "fp8":
        v = v_ref.to(torch.bfloat16).to(torch.float8_e4m3fn)
    else:
        v, sfv = _quant_kmajor(v_ref, pv)
    return q, k, v, sfq, sfk, sfv


def _page_table(batch, seqlen_k, page_size, device="cuda"):
    n = batch * (seqlen_k // page_size)
    return torch.randperm(n, device=device, dtype=torch.int32).reshape(
        batch, seqlen_k // page_size).contiguous()


def _to_pages(x, page_table, page_size):
    b, s, h, d = x.shape
    npages = s // page_size
    out = torch.zeros(int(page_table.max()) + 1, page_size, h, d,
                      device=x.device, dtype=x.dtype)
    out[page_table.reshape(-1).long()] = x.reshape(b, npages, page_size, h, d).reshape(
        b * npages, page_size, h, d)
    return out


def _fwd(*a, **kw):
    o = _flash_attn_fwd(*a, **kw)
    return o[0] if isinstance(o, tuple) else o


def _setup(mode, b, sq, sk, h, d, seed=0, device="cuda"):
    torch.manual_seed(seed)
    q_ref = torch.randn(b, sq, h, d, device=device)
    k_ref = torch.randn(b, sk, h, d, device=device)
    v_ref = torch.randn(b, sk, h, d, device=device)
    q, k, v, sfq, sfk, sfv = _build(mode, q_ref, k_ref, v_ref)
    pt = _page_table(b, sk, PAGE_SIZE, device)
    _, kp, vp, _, sfkp, sfvp = _build(
        mode, q_ref, _to_pages(k_ref, pt, PAGE_SIZE), _to_pages(v_ref, pt, PAGE_SIZE))
    return dict(q=q, k=k, v=v, sfq=sfq, sfk=sfk, sfv=sfv,
                pt=pt, kp=kp, vp=vp, sfkp=sfkp, sfvp=sfvp)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("b,sq,sk,h", [(2, 256, 512, 8), (1, 512, 1024, 4)])
def test_paged_matches_non_paged(mode, causal, b, sq, sk, h):
    t = _setup(mode, b, sq, sk, h, 128)
    ref = _fwd(t["q"], t["k"], t["v"], mSFQ=t["sfq"], mSFK=t["sfk"], mSFV=t["sfv"],
               causal=causal)
    out = _fwd(t["q"], t["kp"], t["vp"], page_table=t["pt"], mSFQ=t["sfq"],
               mSFK=t["sfkp"], mSFV=t["sfvp"], causal=causal)
    torch.cuda.synchronize()
    assert torch.equal(ref, out), (
        f"{mode} paged != non-paged, max diff "
        f"{(ref.float() - out.float()).abs().max().item()}")


@pytest.mark.parametrize("mode", MODES)
def test_paged_seqused_k(mode):
    """Partially filled last page."""
    b, sq, sk, h, d = 3, 256, 512, 8, 128
    t = _setup(mode, b, sq, sk, h, d)
    seqused_k = torch.tensor([sk, sk - 128, sk - 300], dtype=torch.int32, device="cuda")
    ref = _fwd(t["q"], t["k"], t["v"], mSFQ=t["sfq"], mSFK=t["sfk"], mSFV=t["sfv"],
               seqused_k=seqused_k)
    out = _fwd(t["q"], t["kp"], t["vp"], page_table=t["pt"], mSFQ=t["sfq"],
               mSFK=t["sfkp"], mSFV=t["sfvp"], seqused_k=seqused_k)
    torch.cuda.synchronize()
    assert torch.equal(ref, out)


@pytest.mark.parametrize("mode", MODES)
def test_paged_cuda_graph(mode):
    """Capture once, replay after the page table contents change."""
    from flash_attn.cute.interface import flash_attn_func
    b, sq, sk, h, d = 2, 256, 512, 8, 128
    t = _setup(mode, b, sq, sk, h, d)
    pt_a = t["pt"]
    pt_b = _page_table(b, sk, PAGE_SIZE)
    pt = pt_a.clone()

    def call():
        o = flash_attn_func(t["q"], t["kp"], t["vp"], mSFQ=t["sfq"], mSFK=t["sfkp"],
                            mSFV=t["sfvp"], page_table=pt)
        return o[0] if isinstance(o, tuple) else o

    def eager(table):
        pt.copy_(table)
        o = call()
        torch.cuda.synchronize()
        return o.clone()

    ref_a, ref_b = eager(pt_a), eager(pt_b)
    assert not torch.equal(ref_a, ref_b), "page tables must give different results"

    pt.copy_(pt_a)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_buf = call()
    pt.copy_(pt_a); g.replay(); torch.cuda.synchronize()
    assert torch.equal(out_buf, ref_a)
    pt.copy_(pt_b); g.replay(); torch.cuda.synchronize()
    assert torch.equal(out_buf, ref_b)


@pytest.mark.parametrize("page_size", [64, 256])
def test_paged_block_scaled_rejects_bad_page_size(page_size):
    """Block-scaled paged KV only supports page_size == 128; must not be silent."""
    b, sq, sk, h, d = 2, 256, 512, 8, 128
    torch.manual_seed(0)
    q_ref = torch.randn(b, sq, h, d, device="cuda")
    k_ref = torch.randn(b, sk, h, d, device="cuda")
    q, sfq = _quant_tokenmajor(q_ref, True)
    npages = b * (sk // page_size)
    k = torch.zeros(npages, page_size, h, d // 2, device="cuda",
                    dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    v = torch.zeros(npages, page_size, h, d, device="cuda", dtype=torch.bfloat16)
    sfk = torch.zeros(32, 4, h, 4, max(1, d // SF_VEC // 4), 1, npages,
                      device="cuda", dtype=torch.float8_e4m3fn)
    pt = _page_table(b, sk, page_size)
    with pytest.raises(AssertionError, match="page_size == 128"):
        _fwd(q, k, v, page_table=pt, mSFQ=sfq, mSFK=sfk)
