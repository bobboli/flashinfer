# FMHAv2 FP8 PR Clean Report

## Purpose

This note records the cleaned state of PR `#2957` after collapsing the original
debugging history into a minimal commit stack.

This file is intentionally **not** on the PR branch. It lives only on the
backup branch so the PR contains code and tests only.

## Result

The PR branch `fix/fmha-v2-fp8-head256-barrier-deadlock` was force-updated from
the previous 13-commit stack to a 3-commit stack:

1. `6ee5d2c1` `fix(fmha_v2): fix fp8 transpose barrier pipeline on SM90`
2. `c02c2c71` `fix(fmha_v2): fix fp8 persistent scheduler for ragged q-tiles`
3. `4b243593` `test(fmha_v2): enable fp8 output prefill coverage`

## What Changed Relative To The Previous PR Tip

Semantically, the cleaned PR branch matches the previous validated code.

The only tree difference between the old PR tip (`ede6e2e0`) and the cleaned PR
tip (`4b243593`) is:

- deletion of `fp8_hang/REPORT.md`

No kernel code or test logic changed during the cleanup itself.

## Clean Commit Breakdown

### Commit 1: transpose / barrier / scratch-slot correctness

`6ee5d2c1` contains the FP8 warp-specialized kernel fixes:

- CTA-scoped named barrier PTX in `arrive_wait.h`
- FP8 mutex rewrite in `compute.h`
- explicit slot-aware circular-buffer reader helpers in `circular_buffer.h`
- FP8 scratch-buffer depth fix in `kernel_traits.h`
- DMA-side transpose / scratch-slot ordering fixes in `dma.h`

### Commit 2: persistent scheduler fix

`c02c2c71` contains the actual mixed-length scheduler fix:

- exact dynamic `tile_id` decode in `dma.h`
- persistent scheduling kept enabled in `fmha_library.py`

This replaces the earlier temporary static-scheduling workaround.

### Commit 3: fp8-output test enablement

`4b243593` contains the separate fp8-output follow-up:

- remove the stale module-level skip and stale fp8-output skip in
  `test_fmha_v2_prefill.py`
- reduce `kv_tile_buffers` to `1` for the SM90 fp8-output `head_dim > 128` case
  in `fmha_library.py`

This is needed because removing the skip exposed a real H100 shared-memory
budget issue for FP8 output kernels.

## Why Docs Were Removed From The PR

The old PR branch carried `fp8_hang/REPORT.md` as a tracked file. That is useful
for local investigation, but it does not belong in the PR if the goal is a
minimal reviewable patch series.

So the cleanup intentionally removed docs from the PR branch and kept them only
on the backup branch.

## Where The Docs Live

Backup branch:

- `backup/fmha-v2-fp8-head256-barrier-deadlock-20260415`

Relevant docs on the backup branch:

- `fp8_hang/PR_COMMIT_REVIEW.md`
- `fp8_hang/PR_CLEAN_REPORT.md`
- `fp8_hang/REPORT.md`

## Bottom Line

The PR is now in the shape it should have been in from the start:

- one kernel/barrier/scratch fix
- one persistent-scheduler fix
- one fp8-output enablement fix

The debugging notes are preserved, but they are no longer part of the PR
itself.
