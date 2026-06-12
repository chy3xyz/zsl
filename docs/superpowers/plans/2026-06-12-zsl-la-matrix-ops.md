# zsl.la Matrix Operations Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add high-level dense matrix operations (`det`, `inverse_small`, `inverse`, `solve`) to `zsl.la`, aligned with VSL's `la/matrix_ops.v` and `la/densesol.v`.

**Architecture:** Pure-Zig generic functions in `src/la/matrix_ops.zig`, re-exported by `src/la.zig`. Builds on `zsl.lapack.lu`.

**Tech Stack:** Zig 0.17.0-dev, `std.Build`, `std.testing.allocator`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/la/matrix_ops.zig` | `det`, `inverse_small`, `inverse`, `solve`. |
| `src/la.zig` | Re-export `matrix_ops`. |
| `examples/la_ops.zig` | Demo of det/inv/solve. |
| `build.zig` | Add `example-la` step. |
| `README.md`, `AGENTS.md` | Document new operations. |

---

## Task 1: Create `src/la/matrix_ops.zig`

**Files:**
- Create: `src/la/matrix_ops.zig`
- Modify: `src/la.zig`

- [ ] **Step 1: Create file with imports**

```zig
const std = @import("std");
const la = @import("../la.zig");
const lapack = @import("../lapack.zig");
const util = @import("../util.zig");
const Error = @import("../errors.zig").Error;
const Matrix = la.Matrix;
const Vector = la.Vector;
```

- [ ] **Step 2: Update `src/la.zig`**

Add `pub const matrix_ops = @import("la/matrix_ops.zig");` and `test { _ = matrix_ops; }`.

- [ ] **Step 3: Add placeholder test**

```zig
test "matrix_ops placeholder" {
    try std.testing.expect(true);
}
```

- [ ] **Step 4: Run `zig build test`**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/la.zig src/la/matrix_ops.zig
git commit -m "feat(la): add matrix_ops scaffolding"
```

---

## Task 2: Implement `det`

**Files:**
- Modify: `src/la/matrix_ops.zig`

- [ ] **Step 1: Add `det` stub returning `error.ShapeMismatch`**
- [ ] **Step 2: Add failing test** for a 2×2 matrix
- [ ] **Step 3: Run `zig build test`** (expected FAIL)
- [ ] **Step 4: Implement `det`**

Use `dgetrf` and multiply diagonal entries, adjusting sign based on pivot swaps.

- [ ] **Step 5: Run `zig build test`** (expected PASS)
- [ ] **Step 6: Commit**

```bash
git add src/la/matrix_ops.zig
git commit -m "feat(la): add matrix det"
```

---

## Task 3: Implement `inverse_small`

**Files:**
- Modify: `src/la/matrix_ops.zig`

- [ ] **Step 1: Add stub**
- [ ] **Step 2: Add failing tests** for 1×1, 2×2, 3×2 (shape error), 3×3
- [ ] **Step 3: Run `zig build test`** (expected FAIL)
- [ ] **Step 4: Implement `inverse_small`**

Handle 1×1, 2×2, 3×3 explicitly. Return determinant. Error on singular matrix.

- [ ] **Step 5: Run `zig build test`** (expected PASS)
- [ ] **Step 6: Commit**

```bash
git add src/la/matrix_ops.zig
git commit -m "feat(la): add small matrix inverse"
```

---

## Task 4: Implement `inverse`

**Files:**
- Modify: `src/la/matrix_ops.zig`

- [ ] **Step 1: Add stub**
- [ ] **Step 2: Add failing test** for a 4×4 matrix
- [ ] **Step 3: Run `zig build test`** (expected FAIL)
- [ ] **Step 4: Implement `inverse`**

Option A: Use `dgetrf` + repeated `dgetrs` with identity columns.
Option B: Implement `dgetri` (more efficient).

Recommended: Option A for simplicity.

- [ ] **Step 5: Run `zig build test`** (expected PASS)
- [ ] **Step 6: Commit**

```bash
git add src/la/matrix_ops.zig
git commit -m "feat(la): add general matrix inverse"
```

---

## Task 5: Implement `solve`

**Files:**
- Modify: `src/la/matrix_ops.zig`

- [ ] **Step 1: Add stub**
- [ ] **Step 2: Add failing test**
- [ ] **Step 3: Run `zig build test`** (expected FAIL)
- [ ] **Step 4: Implement `solve`**

Copy `b` into `x`, factor `A` with `dgetrf`, then call `dgetrs`.

- [ ] **Step 5: Run `zig build test`** (expected PASS)
- [ ] **Step 6: Commit**

```bash
git add src/la/matrix_ops.zig
git commit -m "feat(la): add linear solve wrapper"
```

---

## Task 6: Add `examples/la_ops.zig`

**Files:**
- Create: `examples/la_ops.zig`
- Modify: `build.zig`

- [ ] **Step 1: Write example**

Demonstrate `det`, `inverse_small`, `inverse`, `solve`.

- [ ] **Step 2: Add `example-la` step to `build.zig`**

- [ ] **Step 3: Run `zig build example-la`**

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add examples/la_ops.zig build.zig
git commit -m "feat(examples): add la_ops demo"
```

---

## Task 7: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Update `README.md`**

Add Phase 5 capability list and `zig build example-la`.

- [ ] **Step 2: Update `AGENTS.md`**

Add `src/la/matrix_ops.zig`, `examples/la_ops.zig` to layout and command list.

- [ ] **Step 3: Commit**

```bash
git add README.md AGENTS.md
git commit -m "docs: document Phase 5 la matrix operations"
```

---

## Task 8: Final verification

- [ ] **Run `zig build test`**
- [ ] **Run `zig build example-la`**
- [ ] **Leak scan**
- [ ] **Commit if any fixes**

---

## Self-Review Checklist

- [ ] `src/la/matrix_ops.zig` exports `det`, `inverse_small`, `inverse`, `solve`.
- [ ] No TODO/TBD placeholders remain.
- [ ] All new files committed.
- [ ] `zig build test` passes.
- [ ] `zig build example-la` passes.
