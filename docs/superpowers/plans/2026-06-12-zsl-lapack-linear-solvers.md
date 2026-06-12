# zsl LAPACK Linear Solvers Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `lapack` module with LU factorization and linear solve routines, aligned with VSL's `lapack64`.

**Architecture:** Pure-Zig generic functions over floating-point types. The LU routines live in `src/lapack/lu.zig` and are re-exported by `src/lapack.zig` and `src/root.zig`.

**Tech Stack:** Zig 0.17.0-dev, `std.Build`, `std.testing.allocator`, `zsl.blas` primitives.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/lapack/lu.zig` | `dgetf2`, `dgetrf`, `dlaswp`, `dgetrs`, `dgesv`. |
| `src/lapack.zig` | Public module re-exports. |
| `src/root.zig` | Adds `_ = lapack;` to the test discovery block. |
| `examples/lapack_solve.zig` | Demo of `dgesv`. |
| `build.zig` | Add `example-lapack` step. |
| `README.md`, `AGENTS.md` | Document new module. |

---

## Task 1: Project structure and public exports

**Files:**
- Create: `src/lapack/lu.zig`
- Create: `src/lapack.zig`
- Modify: `src/root.zig`

- [ ] **Step 1: Create `src/lapack.zig`**

```zig
pub const lu = @import("lapack/lu.zig");
```

- [ ] **Step 2: Create `src/lapack/lu.zig`** with a placeholder test.

```zig
const std = @import("std");

test "lapack placeholder" {
    try std.testing.expect(true);
}
```

- [ ] **Step 3: Update `src/root.zig`**

Add `pub const lapack = @import("lapack.zig");` and `_ = lapack;` inside the test block.

- [ ] **Step 4: Run `zig build test`**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lapack.zig src/lapack/lu.zig src/root.zig
git commit -m "feat(lapack): add module scaffolding"
```

---

## Task 2: Implement `dgetf2` and `dgetrf`

**Files:**
- Modify: `src/lapack/lu.zig`

- [ ] **Step 1: Add `dgetf2` stub returning `false`**
- [ ] **Step 2: Add failing test** for a 3×3 matrix factorization
- [ ] **Step 3: Run `zig build test`** (expected FAIL)
- [ ] **Step 4: Implement `dgetf2`**

Implementation notes:
- Use `zsl.blas.iamax` to find the pivot row.
- Use `zsl.blas.swap` to swap rows.
- Use `zsl.blas.scal` to scale the column below the pivot.
- Use `zsl.blas.ger` to apply the rank-one update.

- [ ] **Step 5: Run `zig build test`** (expected PASS)

- [ ] **Step 6: Implement `dgetrf`** as a wrapper calling `dgetf2`

- [ ] **Step 7: Add tests** for rectangular matrices and singular detection

- [ ] **Step 8: Run `zig build test`** (expected PASS)

- [ ] **Step 9: Commit**

```bash
git add src/lapack/lu.zig
git commit -m "feat(lapack): add dgetf2 and dgetrf"
```

---

## Task 3: Implement `dlaswp` and `dgetrs`

**Files:**
- Modify: `src/lapack/lu.zig`

- [ ] **Step 1: Implement `dlaswp`**

Apply pivot swaps to a matrix.

- [ ] **Step 2: Add `dgetrs` stub returning `error.ShapeMismatch`**
- [ ] **Step 3: Add failing test** for solving a 3×3 system
- [ ] **Step 4: Run `zig build test`** (expected FAIL)
- [ ] **Step 5: Implement `dgetrs`**

Solve using the factored matrix:
- Apply row permutations to `B`.
- Forward solve with `L`.
- Backward solve with `U`.
- For `.trans`, transpose the system before solving.

- [ ] **Step 6: Run `zig build test`** (expected PASS)

- [ ] **Step 7: Commit**

```bash
git add src/lapack/lu.zig
git commit -m "feat(lapack): add dlaswp and dgetrs"
```

---

## Task 4: Implement `dgesv`

**Files:**
- Modify: `src/lapack/lu.zig`

- [ ] **Step 1: Add `dgesv` stub**
- [ ] **Step 2: Add failing test**
- [ ] **Step 3: Run `zig build test`** (expected FAIL)
- [ ] **Step 4: Implement `dgesv`** as `dgetrf` + `dgetrs`
- [ ] **Step 5: Run `zig build test`** (expected PASS)
- [ ] **Step 6: Commit**

```bash
git add src/lapack/lu.zig
git commit -m "feat(lapack): add dgesv"
```

---

## Task 5: Add `examples/lapack_solve.zig`

**Files:**
- Create: `examples/lapack_solve.zig`
- Modify: `build.zig`

- [ ] **Step 1: Write example**

Solve `A·X = B` for a small dense system and print the solution.

- [ ] **Step 2: Add `example-lapack` step to `build.zig`**

- [ ] **Step 3: Run `zig build example-lackack`**

Expected: PASS and print results.

- [ ] **Step 4: Commit**

```bash
git add examples/lapack_solve.zig build.zig
git commit -m "feat(examples): add lapack_solve demo"
```

---

## Task 6: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Update `README.md`**

Add Phase 4 capability list and `zig build example-lapack`.

- [ ] **Step 2: Update `AGENTS.md`**

Add `src/lapack.zig`, `src/lapack/lu.zig`, `examples/lapack_solve.zig` to layout.

- [ ] **Step 3: Commit**

```bash
git add README.md AGENTS.md
git commit -m "docs: document Phase 4 LAPACK linear solvers"
```

---

## Task 7: Final verification

- [ ] **Run `zig build test`**
- [ ] **Run `zig build example-lapack`**
- [ ] **Leak scan** — verify no new allocations without `errdefer`/`defer`
- [ ] **Commit if any fixes**

---

## Self-Review Checklist

- [ ] `src/lapack.zig` and `src/lapack/lu.zig` exist.
- [ ] `dgetf2`, `dgetrf`, `dlaswp`, `dgetrs`, `dgesv` exported.
- [ ] No TODO/TBD placeholders remain.
- [ ] All new files committed.
- [ ] `zig build test` passes.
- [ ] `zig build example-lapack` passes.
