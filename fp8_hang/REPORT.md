# FMHAv2 FP8 Deadlock Report

## Scope

This note summarizes the FP8 `fmha_v2` deadlock investigation on Hopper SM90
for the warp-specialized QGMMA path. It is based on the current branch state,
current sanitizer runs, and direct repro harnesses in this workspace.

Target configuration:

- Input layout: `PACKED_QKV`
- Dtype: FP8 `e4m3`
- Output dtype: BF16
- Mask: causal
- Head dim: `256`
- Heads: `num_qo_heads=32`, `num_kv_heads=4`
- GPU: H100 / SM90

## Final Status

The deadlock is resolved on the current local branch.

There were two separate correctness problems:

1. An FP8 transpose scratch-buffer / barrier bug in the DMA path.
2. A persistent-scheduler bug for mixed-length launches, where the scheduler
   linearized work using a uniform `num_tiles_per_head` derived from the maximum
   sequence length and then skipped invalid tiles later.

The current branch fixes both. FP8 kernels are back on persistent scheduling,
and the mixed-length reproducer that previously deadlocked under `racecheck` is
now clean.

## Root Cause

The remaining deadlock after the earlier barrier cleanup was not "FP8 hd256" in
general. The failure boundary was mixed per-sequence q-tile counts under the
persistent scheduler.

The old dynamic scheduler used one global `num_tiles_per_head`:

- `num_tiles_per_head = ceil(max_seqlen / (STEP_Q * NUM_COMPUTE_GROUPS))`
- `tile_id -> (batch, head, q_step_offset)` was decoded arithmetically from
  that uniform tile space
- shorter sequences produced tile ids whose `q_tile_offset` landed past
  `actual_seqlen`
- those invalid tiles were skipped in the DMA loop

That skip model was tolerated by the BF16/FP16 path, but not by the FP8
warp-specialized QGMMA path with explicit V transpose. Under `racecheck`, the
mixed-length launch stalled on the Q/K/V scratch trackers because the producer
and consumer pipelines no longer advanced in lockstep.

## Code Changes

### 1. FP8 transpose / barrier fixes

The branch keeps the earlier FP8-specific fixes:

- `V_SCRATCH_BUFFERS` now matches the KV double-buffer depth.
- `CircularBufferReader` supports explicit slot-based wait / pop operations.
- `transpose_v_tile()` waits on and pops the actual scratch slot produced by
  `load_kv()`.
- the V-scratch TMA barrier pointer is read only after the DMA-group sync in
  `PREPARE_KV_BUFFER()`.
- CTA named barriers use `barrier.cta.arrive` / `barrier.cta.sync`.
- the FP8 compute mutex uses shared-memory mbarrier `arrive()/wait()`.

These changes removed the earlier timing-sensitive scratch-slot reuse bug.

### 2. Exact dynamic tile decode for FP8 persistent scheduling

The final fix is in the DMA scheduler:

- for dynamic scheduling on the FP8 transpose path
  (`DMA_GROUP_TRANSPOSE_V == true`), the DMA warp no longer decodes `tile_id_`
  from a uniform `num_tiles_per_head`
- instead, it walks `cu_q_seqlens` directly and decodes `tile_id_` against the
  exact number of valid q-tile groups for each batch element
- once `tile_id_` exceeds the exact valid work count, the DMA loop exits rather
  than scheduling a fake tile and skipping it later

This preserves persistent scheduling while removing the invalid-tile pacing
hazard that was specific to mixed-length FP8 launches.

### 3. Re-enable persistent scheduling for FP8

`fmha_library.py` now puts FP8 kernels back on `scheduling_mode = 1`.

The persistent scheduler is no longer being avoided. The fix is in the scheduler
decode itself.

## Validation

Validated locally on the current tree after clearing the JIT cache for the FP8
module.

### PACKED_QKV, FP8, mixed lengths `[1024, 514]`

- unsanitized: clean
- unsanitized stress loop: 20/20 pass
- `compute-sanitizer --tool racecheck --racecheck-deadlock-timeout 10000`:
  clean
- `compute-sanitizer --tool synccheck`: clean

### PACKED_QKV, FP8, ragged `batch_size=16`

- unsanitized: clean
- `compute-sanitizer --tool racecheck --racecheck-deadlock-timeout 10000`:
  clean
- `compute-sanitizer --tool synccheck`: clean

### PACKED_QKV, FP8, uniform `batch_size=16`, all `1024`

- `compute-sanitizer --tool racecheck --racecheck-deadlock-timeout 10000`:
  clean

### CONTIGUOUS_Q_KV, FP8, mixed lengths `[1024, 514]`

- unsanitized: clean
- `compute-sanitizer --tool racecheck --racecheck-deadlock-timeout 10000`:
  clean
- `compute-sanitizer --tool synccheck`: clean

### FP16 control

- uniform FP16 control remains clean under
  `compute-sanitizer --tool racecheck --racecheck-deadlock-timeout 120000`

## Files Most Relevant To The Fix

- `csrc/fmha_v2/fmha/warpspec/dma.h`
- `csrc/fmha_v2/fmha/warpspec/circular_buffer.h`
- `csrc/fmha_v2/fmha/warpspec/kernel_traits.h`
- `csrc/fmha_v2/fmha/hopper/arrive_wait.h`
- `csrc/fmha_v2/fmha/warpspec/compute.h`
- `flashinfer/jit/attention/fmha_v2/fmha_library.py`

## Remaining Work

The correctness issue is resolved, but two follow-ups remain outside the kernel
logic itself:

1. The branch is not PR-clean yet. The working tree still contains unrelated
   untracked investigation artifacts in this workspace.
2. I have not yet measured whether the exact FP8 dynamic decode changes
   throughput relative to the old arithmetic scheduler on ragged batches.

## Bottom Line

The final fix is not a static-scheduling fallback. The FP8 persistent scheduler
is active again.

What changed is the scheduler's work decode: FP8 dynamic scheduling now uses the
exact valid q-tile count from `cu_q_seqlens` instead of iterating a max-length
tile space and skipping invalid tiles later. That removes the mixed-length
deadlock while keeping the persistent scheduler enabled.
