"""Curated seed knowledge for the library.

A starter set of *transferable* Blackwell / Triton optimization lessons, so a
fresh machine's first run isn't cold. These are paraphrased in our own words
(the underlying techniques are facts, not protected expression) and attributed
to their source.

Source: https://github.com/Dogacel/auto-gpu-kernel (Apache-2.0, © Doğaç Eldenk) —
the winning agent-only entry of the MLSys 2026 FlashInfer kernel-generation
contest. We lift only the general, problem-independent findings; the
sparse-attention-specific ones are left out. See the repo NOTICE for attribution.

Populate with ``kernel-harness seed-library`` (or ``seed_library(lib)``).
"""

from __future__ import annotations

from .library import Library, LibraryEntry

_SOURCE = {"source": "github.com/Dogacel/auto-gpu-kernel", "license": "Apache-2.0"}


def _tech(title: str, text: str, tags: list[str]) -> LibraryEntry:
    return LibraryEntry("technique", title, f"[technique] {title}: {text}", _SOURCE | {"tags": tags})


def _hw(title: str, text: str, tags: list[str]) -> LibraryEntry:
    return LibraryEntry("hardware_note", title, f"[hardware] {title}: {text}", _SOURCE | {"tags": tags})


# Transferable lessons (paraphrased). Tag with the GPU family / tool they apply to.
SEED_ENTRIES: list[LibraryEntry] = [
    _tech(
        "split-K / flash-decoding for SM-starved kernels",
        "When the launch grid has far fewer CTAs than the GPU has SMs (e.g. one CTA "
        "per token on a few tokens), most SMs sit idle. Split the reduction axis "
        "across NUM_SPLITS programs and combine — often the single biggest structural "
        "win on low-parallelism workloads.",
        ["triton", "attention", "occupancy"],
    ),
    _tech(
        "a scalar `if` in the hot loop defeats num_stages prefetching",
        "Gating the loop body with a scalar-conditioned `if` stops Triton from issuing "
        "async loads for iteration i+1 while computing i. Keep compute unconditional: "
        "precompute a dynamic loop bound before the loop (range(0, dyn_upper, STEP)) or "
        "mask with tl.where inside the tile ops. Triton also rejects `break` in for-loops.",
        ["triton", "pipelining"],
    ),
    _tech(
        "static_range unroll vs dynamic range",
        "tl.static_range(N) inlines the body N times — great ILP for small N and small "
        "per-iter tiles, but blows up register/icache pressure at large N (>=16) or large "
        "tiles (>=32 KB/iter), causing big regressions. Use dynamic range()+num_stages>1 "
        "for large loops; keep static_range for small, regular reductions.",
        ["triton", "pipelining"],
    ),
    _tech(
        "online-softmax NaN landmine on fully-masked tiles",
        "exp2(-inf - -inf) = NaN. Any online softmax over a subset that might be fully "
        "masked must guard: m_safe = tl.where(m == -inf, 0.0, m) before exp2, and "
        "alpha = tl.where(m_prev == -inf, 0.0, exp2(m_prev - m_safe)).",
        ["triton", "softmax", "numerics"],
    ),
    _tech(
        "use base-2 LSE / exp2 in softmax",
        "Compute log-sum-exp in base 2 (lse = m + log2(l)) and premultiply the softmax "
        "scale by log2e (1.4427) so logits land in log2-space and softmax can use the "
        "faster exp2.",
        ["triton", "softmax", "numerics"],
    ),
    _tech(
        "don't pre-scale bf16 via cast->mul->cast",
        "Scaling a bf16 tensor by going bf16->fp32->bf16 truncates the 7-bit mantissa and "
        "the error compounds over long dot products. Apply the scale post-dot on the fp32 "
        "logits/accumulator instead.",
        ["triton", "numerics", "bf16"],
    ),
    _tech(
        "host-side input-characteristic dispatch is free",
        "Branching on an input shape on the host (e.g. `if num_tokens <= 2:`) costs nothing "
        "— each branch loads one compiled kernel variant. Specializing per workload regime "
        "can beat a one-size-fits-all kernel by a wide margin. (But you cannot cheaply "
        "dispatch on per-element/on-device scalars from the host.)",
        ["triton", "dispatch"],
    ),
    _tech(
        ".item() in a kernel launcher hot path is catastrophic",
        "A host<-device sync (e.g. int(x.sum().item())) in the launch path serializes the "
        "GPU queue and adds tens of microseconds even for a 'cheap' reduction. Keep "
        "data-dependent decisions on-device, or dispatch only on host-known shapes.",
        ["triton", "dispatch", "latency"],
    ),
    _tech(
        "persist/reuse scratch buffers across calls",
        "Avoid torch.empty/zeros per call for fixed (shape,dtype) scratch by lazily "
        "allocating once and reusing — as long as contents are fully recomputed each call. "
        "Never cache *results* (that's benchmark gaming).",
        ["pytorch", "latency"],
    ),
    _tech(
        "exploit trivializing input shapes",
        "Check whether the input shape makes the op trivial: softmax over a size-1 axis is "
        "1.0; a reduction over a size-1 dim is a no-op; attention with seq-len 1 returns the "
        "value vector; gather with k<=N is an index-select. Optimizing the common easy case "
        "of a skewed distribution gives outsized wins.",
        ["general", "specialization"],
    ),
    _tech(
        "fuse launches and avoid .contiguous() copies",
        "Fold prologue/epilogue ops (init, padding, masking, remapping) into the main kernel "
        "instead of separate launches, and plumb strides into the kernel instead of calling "
        ".contiguous() — each copy is both a launch and a memory round-trip.",
        ["triton", "fusion"],
    ),
    _tech(
        "atomic cross-CTA barrier to fuse serial kernels",
        "Two serial kernels can be fused into one launch with a release/acquire atomic "
        "barrier: atomic_add(+1, sem='release') to signal, spin on a load until count==N to "
        "wait. A monotonic generation counter beats a reset-decrement; a volatile load beats "
        "atomic_add(0) for the spin. Only safe when all CTAs fit on the SMs (no deadlock).",
        ["triton", "fusion", "advanced"],
    ),
    _hw(
        "Blackwell: num_warps=8 is a strong default for H=16 MLA-style kernels",
        "Confirmed in both directions — num_warps=4 serializes across heads, num_warps=16 "
        "wastes MMA tile rows (MMA tile is [16,16,16] minimum). Don't sweep num_warps on this "
        "shape unless H or the MMA shape changes.",
        ["blackwell", "triton", "tuning"],
    ),
    _hw(
        "Blackwell: bf16xbf16 tl.dot ignores input_precision",
        "There is a single PTX instruction for bf16xbf16->fp32 on sm_100 (wgmma.bf16.bf16.f32), "
        "so input_precision ('ieee'/'tf32x3') is a no-op for bf16 dots. For fp32 dots, try "
        "tf32x3 first and fall back to ieee only if abs_err exceeds tolerance.",
        ["blackwell", "triton", "numerics"],
    ),
    _hw(
        "Blackwell: Gluon dot_fma has no tensor cores",
        "gl.dot_fma is pure software FMA (~60-80x slower than tl.dot on B200) — a correctness "
        "scaffold only. For tensor-core throughput in Gluon you must use Blackwell primitives "
        "(bw.tcgen05_mma with tensor-memory descriptors), which are hard to make competitive "
        "below large effective tile sizes.",
        ["blackwell", "gluon", "advanced"],
    ),
    _hw(
        "Blackwell: L2 cache-residency vs HBM-reduction headroom",
        "B200 has ~126 MB L2. Short-lived producer/consumer intermediates (e.g. split->combine) "
        "stay L2-resident, so halving their byte volume doesn't help — verify bytes actually hit "
        "HBM before doing dtype reduction. 'Near HBM floor' claims are byte-volume-dependent: "
        "a structural rewrite (e.g. more splits) can move the floor.",
        ["blackwell", "memory"],
    ),
    _hw(
        "Blackwell: cache_modifier and L2 eviction hints are per-kernel levers",
        "cache_modifier='.cg' (bypass L1) helps when the grid is SM-starved and lines may be "
        "re-touched, but hurts when the grid is full — apply per-kernel, not globally. "
        "eviction_policy is a separate L2-replacement hint: 'evict_first' for genuinely one-shot "
        "loads that free >=tens of KB; 'evict_last' for multi-reader cross-CTA shared loads. They "
        "compose on loads, but ptxas rejects .cg+evict on stores.",
        ["blackwell", "memory", "advanced"],
    ),
    _tech(
        "measurement discipline: absolute latency, one change per iter, A/B for small deltas",
        "Trust absolute latencies over speedup ratios (ratios are noisy across machines). Make "
        "one optimization per iteration so wins are attributable. For sub-5% deltas, confirm with "
        "a paired A/B re-benchmark on the same machine. Report latency split by workload regime "
        "(small vs large) — aggregate means hide regime-specific regressions.",
        ["methodology"],
    ),
    _tech(
        "judge a direction by its ceiling, not its iteration-0 number",
        "A fresh approach at iteration 1 is slower than a mature one at iteration 20. Evaluate the "
        "potential ceiling of each direction; pivot when the current approach's ceiling is below "
        "an alternative's, even if early numbers look bad.",
        ["methodology", "strategy"],
    ),
]


def seed_library(library: Library) -> int:
    """Add the seed entries to ``library`` (idempotent). Returns count added/refreshed."""
    for entry in SEED_ENTRIES:
        library.add(entry)
    return len(SEED_ENTRIES)
