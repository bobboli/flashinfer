# FMHAv2 FP8 PR Commit Review

## Scope

This note reviews the current PR branch history for the FMHAv2 FP8 deadlock work.

- Branch: `fix/fmha-v2-fp8-head256-barrier-deadlock`
- Reviewed range: `19b825d36134e350ce63faa6e3a549881a58e423..ede6e2e05009590a2fba8878d4a2d6a754a5d574`
- Commits in range: 13, including 1 merge commit

The goal is not to repeat the debugging narrative. The goal is to answer a narrower question:

Which commits are actually required for the final PR, and which commits are only investigation churn, temporary workarounds, or superseded attempts?

This review is based on:

- the current `HEAD` tree
- per-commit diffs in the branch
- the final validated behavior on this branch: manual repros, `compute-sanitizer` runs, and the focused fp8-output test matrix

## Executive Summary

The current branch contains three real fixes buried under a large amount of intermediate churn:

1. FP8 transpose / barrier / scratch-slot correctness fixes in the warp-specialized kernel.
2. A real persistent-scheduler fix for mixed-length FP8 launches.
3. A separate SM90 shared-memory budget fix for FP8 output kernels, plus removal of a stale test skip.

Everything else is either:

- temporary test skipping
- timeout scaffolding
- an intermediate workaround that was later replaced
- iterative rework of an earlier attempt
- merge noise

If this PR were cleaned up, it should be squashed to roughly:

1. `fix(fmha_v2): fix fp8 transpose/barrier pipeline on SM90`
2. `fix(fmha_v2): fix fp8 persistent scheduler for ragged q-tile counts`
3. `test(fmha_v2): enable fp8 output prefill coverage and fit SM90 smem budget`

The existing `fp8_hang/REPORT.md` is useful as a local investigation note, but it is not required for code correctness and does not need to be part of the PR if the goal is a minimal patch series.

## What The Final Tree Actually Needs

The final code changes that appear necessary are:

- `csrc/fmha_v2/fmha/hopper/arrive_wait.h`
  - use CTA-scoped named-barrier PTX via `barrier.cta.arrive` / `barrier.cta.sync`
- `csrc/fmha_v2/fmha/warpspec/compute.h`
  - replace the FP8 compute-side predicated named-barrier mutex path with `mutex.arrive(); mutex.wait();`
- `csrc/fmha_v2/fmha/warpspec/circular_buffer.h`
  - add slot-specific `peek/wait/advance/pop` overloads
- `csrc/fmha_v2/fmha/warpspec/kernel_traits.h`
  - make `V_SCRATCH_BUFFERS` reuse `KV_BUFFERS` on the FP8 transpose path
- `csrc/fmha_v2/fmha/warpspec/dma.h`
  - keep `UNROLL=1` for `STEP_KV >= 128`
  - keep the DMA-group sync after `cbw_v.threadReserve()`
  - reserve/read the FP8 scratch-slot barrier pointer after DMA-group rendezvous
  - wait/pop the actual scratch slot that was produced
  - decode FP8 dynamic `tile_id_` using exact valid q-tile counts from `cu_q_seqlens`
- `flashinfer/jit/attention/fmha_v2/fmha_library.py`
  - keep FP8 on persistent scheduling now that exact tile decode exists
  - reduce `kv_tile_buffers` to `1` for SM90 fp8-output `head_dim > 128`
- `tests/attention/test_fmha_v2_prefill.py`
  - remove the blanket fp8-output skip

Those are the real final-state changes. The rest of the history is just how the branch got there.

## Commit-by-Commit Review

| Commit | Topic | Necessary In Final PR? | Review |
| --- | --- | --- | --- |
| `15598df4` | Narrow module-level skip into per-case skips | No | Investigation scaffolding only. It replaced an overly broad skip with narrower skips, but it still kept the branch in workaround mode rather than fixing the kernel. This should be dropped in a squash. |
| `abaad51f` | Move FP8 `head_dim=256` skip into helper | No | Same temporary workaround, just moved to a better location. No real fix. Drop. |
| `56bebe64` | Add DMA-group sync after `cbw_v.threadReserve()` | Yes, but not as a standalone commit | This hunk survives in the final tree and still looks necessary. It prevents DMA warps from splitting around the destination-slot reserve in `transpose_v_tile()`. The commit message overclaimed the effect, because this did not fully solve the deadlock by itself. Keep the hunk, not the commit boundary. |
| `4a775fb3` | Remove `fp8 + head_dim=256` skip | Yes in spirit, but not as this commit | The final tree should indeed stop skipping the hd256 cases. But at this point in history the branch was not actually done yet. Fold the net effect into the final test-enablement commit, not as an early standalone “fixed now” commit. |
| `07ad9544` | Add `pytest-timeout` and module timeout marker | No | Temporary harness workaround. It was later removed and is not part of the final tree. Drop. |
| `568c21b9` | Adjust timeout comment | No | Comment churn on a temporary workaround that is already gone. Drop. |
| `81a6653d` | First combined kernel fix attempt | Partially | This commit introduced two ideas that still survive: `UNROLL=1` for `STEP_KV >= 128` and the FP8 compute mutex rewrite. But its named-barrier PTX change was still in flux, and the branch was not actually clean after this point. Keep only the surviving hunks via squashing. |
| `0c26b04e` | Rework of the previous attempt | No as a separate commit | This mostly serves as history cleanup: it backed out the timeout scaffolding and reverted the earlier named-barrier spelling change while preserving the mutex direction. There is no reason to keep this as a separate PR commit after squashing. |
| `e3b56c4d` | Refined combined kernel fix attempt | Partially | This is the first version whose `compute.h` change matches the final fp8 mutex rewrite. It also keeps the `UNROLL=1` change. But it still was not sufficient for the actual racecheck issue, so it should be folded into the broader kernel-fix squash. |
| `4fe4e7c1` | Static-scheduler fallback plus scratch-slot fixes | Partially, and this is an important one | The static scheduling fallback was only a containment workaround and is superseded by the later persistent-scheduler fix. However, the scratch-slot lifetime fixes in `circular_buffer.h`, `kernel_traits.h`, `dma.h`, and the CTA barrier PTX in `arrive_wait.h` survive and appear necessary. Keep those hunks. Drop the static-scheduling part. |
| `880661e9` | Merge commit | No | Pure history noise. Rebase it away. |
| `19387ee0` | Exact FP8 persistent-scheduler tile decode | Yes | This is the real fix for the remaining mixed-length FP8 deadlock. It keeps persistent scheduling enabled, but removes the invalid-tile pacing hazard by decoding `tile_id_` against exact valid q-tile counts from `cu_q_seqlens`. This commit should survive, either standalone or as one dedicated squash. |
| `ede6e2e0` | Enable fp8-output prefill tests | Yes | This is a real follow-up fix, not churn. Removing the stale fp8-output skip exposed a separate SM90 shared-memory budget failure for FP8 output `head_dim=256`. The `fmha_library.py` adjustment to `kv_tile_buffers` is required, and the test skip removal should stay. |

## Detailed Notes On The Important Commits

### `56bebe64` is necessary but not sufficient

This commit added the DMA-group sync after `cbw_v.threadReserve()` in `transpose_v_tile()`.

That synchronization is still present in `HEAD`, and the branch still needs it. Without it, the DMA producer side can split around the destination-slot reserve and phase-flip the consumed barrier under slower warps.

What makes this commit misleading is not the change itself. The problem is that the commit treated that one fix as the whole story. Later investigation showed that the branch still had:

- scratch-slot reuse problems
- sanitizer-invalid named-barrier usage
- a separate persistent-scheduler bug on mixed-length FP8 launches

Conclusion: keep the hunk, but do not preserve the commit as “the deadlock fix.”

### `81a6653d` / `0c26b04e` / `e3b56c4d` should collapse into one clean kernel-fix commit

These three commits are iterations on the same theme. The final tree only needs the stable end result:

- keep the FP8 `compute.h` mutex rewrite
- keep the `STEP_KV >= 128` transposer unroll reduction
- keep the final CTA-scoped named-barrier PTX spelling
- drop the timeout cleanup noise from this section of history

In other words, the branch does need the kernel changes that emerged from this iteration cycle, but not the iteration cycle itself.

### `4fe4e7c1` mixes one real fix with one temporary workaround

This commit is structurally important because it introduced the scratch-buffer changes that survived:

- `V_SCRATCH_BUFFERS = KV_BUFFERS`
- explicit scratch-slot wait/pop APIs
- slot-specific use of the scratch tracker in `transpose_v_tile()`
- ordering fix for reading the scratch barrier pointer
- CTA-scoped named-barrier PTX

Those are real final changes.

The static-scheduler fallback in `fmha_library.py` is not. That fallback was a useful containment step when `racecheck` still showed mixed-length persistent-scheduler deadlocks, but it was explicitly replaced by `19387ee0`.

Conclusion: split this commit conceptually into:

- kernel/scratch fixes to keep
- static-scheduling workaround to drop

### `19387ee0` is the key scheduler commit

This commit is the strongest standalone candidate to keep intact.

It is the commit that turns the “fp8 must use static scheduling” workaround into an actual scheduler fix:

- for the FP8 transpose path, dynamic scheduling no longer decodes work from a uniform `num_tiles_per_head`
- instead, it walks `cu_q_seqlens` and computes the exact number of valid q-tile groups per batch item
- invalid tile ids are never scheduled in the first place

That is the actual resolution of the mixed-length persistent-scheduler deadlock.

### `ede6e2e0` is a real second bug fix, not cleanup

This commit should not be dismissed as “just deleting a skip.”

Removing the fp8-output skip surfaced a separate real failure:

- SM90 FP8-output `head_dim=256` kernels exceeded H100’s dynamic shared-memory budget
- kernel launch setup failed with `cudaFuncSetAttribute(...): invalid argument`

The fix in `fmha_library.py` is real:

- preserve `kv_loop_step = 128`
- reduce `kv_tile_buffers` to `1` only for the fp8-output `head_dim > 128` case

This change is necessary if the PR is going to claim fp8-output test coverage rather than merely stop skipping it.

## Commits That Can Be Dropped Safely

The following commits should disappear entirely in a cleaned-up PR history:

- `15598df4`
- `abaad51f`
- `07ad9544`
- `568c21b9`
- `880661e9`

The following commits should not remain as separate commits, even though some of their hunks should survive in squashed form:

- `56bebe64`
- `4a775fb3`
- `81a6653d`
- `0c26b04e`
- `e3b56c4d`
- `4fe4e7c1`

The following commits are strong candidates to survive as separate logical commits after an interactive rebase:

- `19387ee0`
- `ede6e2e0`

## Recommended Squash Plan

### Commit 1: kernel/barrier/scratch correctness

This should contain the final contents of:

- `csrc/fmha_v2/fmha/hopper/arrive_wait.h`
- `csrc/fmha_v2/fmha/warpspec/compute.h`
- `csrc/fmha_v2/fmha/warpspec/circular_buffer.h`
- `csrc/fmha_v2/fmha/warpspec/kernel_traits.h`
- the non-scheduler parts of `csrc/fmha_v2/fmha/warpspec/dma.h`

Sources to fold into it:

- `56bebe64`
- the surviving parts of `81a6653d`
- the surviving parts of `e3b56c4d`
- the surviving non-static-scheduler parts of `4fe4e7c1`

### Commit 2: persistent scheduler fix

This should mostly be `19387ee0`:

- exact dynamic tile decode in `dma.h`
- persistent scheduling re-enabled in `fmha_library.py`

### Commit 3: fp8-output enablement

This should mostly be `ede6e2e0`, optionally folding the spirit of `4a775fb3` into it:

- remove stale fp8-output skip
- reduce `kv_tile_buffers` for SM90 fp8-output `head_dim > 128`

### Optional commit 4: documentation only

If the PR wants to keep an investigation note, keep a docs-only commit for:

- `fp8_hang/REPORT.md`

If the PR wants to stay minimal, leave this out of the PR and keep it local.

## Bottom Line

The branch does not need most of its current commit boundaries.

What it does need is the final substance:

- one cleaned-up kernel/barrier/scratch fix
- one real persistent-scheduler fix
- one fp8-output enablement fix

That is the defensible PR. Everything else should be treated as intermediate history and squashed away.
