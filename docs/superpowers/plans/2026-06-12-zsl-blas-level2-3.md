# zsl BLAS Level-2/3 Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Extend `src/blas.zig` with BLAS Level-2 (`gemv`, `ger`) and Level-3 (`gemm`) operations, with passing tests and a runnable `matrix_ops` example.

**Architecture:** Pure-Zig generic functions over floating-point types, operating on `Vector(T)` and `Matrix(T)`. A small `src/blas/types.zig` module provides the `Transpose` enum.

**Tech Stack:** Zig 0.17.0-dev, `std.Build`, `std.testing.allocator`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/blas/types.zig` | `Transpose` enum. |
| `src/blas.zig` | Adds `gemv`, `ger`, `gemm` and their tests. Re-exports `Transpose`. |
| `src/root.zig` | Updated to reference `src/blas/types.zig` tests. |
| `examples/matrix_ops.zig` | Demo of `gemv` and `gemm`. |

---

## Task 1: Add `Transpose` enum

**Files:**
- Create: `src/blas/types.zig`
- Modify: `src/blas.zig` (import and re-export)
- Modify: `src/root.zig` (reference tests)

- [ ] **Step 1: Create `src/blas/types.zig`**

```zig
pub const Transpose = enum {
    no_trans,
    trans,
    conj_trans,
};

test "Transpose variants exist" {
    const t: Transpose = .trans;
    try @import("std").testing.expect(t == .trans);
}
```

- [ ] **Step 2: Update `src/blas.zig`**

Add near the top:

```zig
pub const types = @import("blas/types.zig");
pub const Transpose = types.Transpose;
```

- [ ] **Step 3: Update `src/root.zig` test block**

Add `_ = blas.types;` inside the existing `test { ... }` block so tests in `src/blas/types.zig` are discovered.

- [ ] **Step 4: Run `zig build test`**

Expected: PASS (only the new enum test runs).

- [ ] **Step 5: Commit**

```bash
git add src/blas/types.zig src/blas.zig src/root.zig
git commit -m "feat(blas): add Transpose enum"
```

---

## Task 2: Implement `gemv`

**Files:**
- Modify: `src/blas.zig`

- [ ] **Step 1: Add `gemv` stub that returns `error.ShapeMismatch`**

```zig
pub fn gemv(
    comptime T: type,
    trans_a: Transpose,
    alpha: T,
    a: Matrix(T),
    x: Vector(T),
    beta: T,
    y: *Vector(T),
) Error!void {
    _ = util.Float(T);
    _ = trans_a;
    _ = alpha;
    _ = a;
    _ = x;
    _ = beta;
    _ = y;
    return error.ShapeMismatch;
}
```

- [ ] **Step 2: Add a failing test**

```zig
test "gemv no_trans f64" {
    const T = f64;
    const V = Vector(T);
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 2, 3, &[_]T{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer a.deinit(std.testing.allocator);
    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 0.5, 2.0 });
    defer x.deinit(std.testing.allocator);
    var y = try V.fromSlice(std.testing.allocator, &[_]T{ 0.0, 0.0 });
    defer y.deinit(std.testing.allocator);

    try gemv(T, .no_trans, 1.0, a, x, 0.0, &y);
    const float = @import("float.zig");
    try std.testing.expect(float.approxEqAbs(T, try y.get(0), 8.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try y.get(1), 17.0, 1e-12));
}
```

- [ ] **Step 3: Run `zig build test`**

Expected: FAIL — gemv returns `ShapeMismatch`.

- [ ] **Step 4: Implement `gemv`**

Replace the stub with full implementation handling `.no_trans` and `.trans`.

- [ ] **Step 5: Run `zig build test`**

Expected: PASS.

- [ ] **Step 6: Add additional tests**

- `.trans` case for `f64`
- `beta = 1` accumulation case
- shape mismatch case

- [ ] **Step 7: Run `zig build test`**

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/blas.zig
git commit -m "feat(blas): add gemv"
```

---

## Task 3: Implement `ger`

**Files:**
- Modify: `src/blas.zig`

- [ ] **Step 1: Add `ger` stub returning `ShapeMismatch`**
- [ ] **Step 2: Add a failing test for rank-one update**
- [ ] **Step 3: Run `zig build test`** (expected FAIL)
- [ ] **Step 4: Implement `ger`**
- [ ] **Step 5: Run `zig build test`** (expected PASS)
- [ ] **Step 6: Add shape mismatch test**
- [ ] **Step 7: Run `zig build test`** (expected PASS)
- [ ] **Step 8: Commit**

```bash
git add src/blas.zig
git commit -m "feat(blas): add ger"
```

---

## Task 4: Implement `gemm`

**Files:**
- Modify: `src/blas.zig`

- [ ] **Step 1: Add `gemm` stub returning `ShapeMismatch`**
- [ ] **Step 2: Add a failing test for no_trans × no_trans**
- [ ] **Step 3: Run `zig build test`** (expected FAIL)
- [ ] **Step 4: Implement simple `gemm`**

Start with a triple-loop implementation supporting all four transpose combinations.

- [ ] **Step 5: Run `zig build test`** (expected PASS)

- [ ] **Step 6: Add blocked algorithm**

When both `trans_a` and `trans_b` are `.no_trans` and the inner dimension exceeds a comptime block size (e.g., 32), use a blocked nested loop to improve cache reuse.

- [ ] **Step 7: Add tests**

- All four transpose combinations (`nn`, `nt`, `tn`, `tt`) for `f64`
- `beta = 0`, `beta = 1`, `beta = 0.5`
- Shape mismatch cases
- A larger matrix to exercise blocking path

- [ ] **Step 8: Run `zig build test`** (expected PASS)

- [ ] **Step 9: Commit**

```bash
git add src/blas.zig
git commit -m "feat(blas): add gemm with blocking"
```

---

## Task 5: Add `examples/matrix_ops.zig`

**Files:**
- Create: `examples/matrix_ops.zig`

- [ ] **Step 1: Write the example**

```zig
const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const T = f64;
    const V = zsl.la.Vector(T);
    const M = zsl.la.Matrix(T);

    var a = try M.fromRowSlice(allocator, 2, 3, &[_]T{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer a.deinit(allocator);
    var x = try V.fromSlice(allocator, &[_]T{ 1.0, 0.5, 2.0 });
    defer x.deinit(allocator);
    var y = try V.fromSlice(allocator, &[_]T{ 0.0, 0.0 });
    defer y.deinit(allocator);

    try zsl.blas.gemv(T, .no_trans, 1.0, a, x, 0.0, &y);
    std.debug.print("gemv(A, x) = {any}\n", .{y.rawData()});

    var b = try M.fromRowSlice(allocator, 3, 2, &[_]T{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    });
    defer b.deinit(allocator);
    var c = try M.init(allocator, 2, 2);
    defer c.deinit(allocator);

    try zsl.blas.gemm(T, .no_trans, .no_trans, 1.0, a, b, 0.0, &c);
    std.debug.print("gemm(A, B) = {any}\n", .{c.rawData()});
}
```

- [ ] **Step 2: Run `zig build example`**

Expected: PASS and print results.

- [ ] **Step 3: Commit**

```bash
git add examples/matrix_ops.zig
git commit -m "feat(examples): add matrix_ops demo"
```

---

## Task 6: Update `AGENTS.md` and `README.md`

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`

- [ ] **Step 1: Update `AGENTS.md`**

Add `src/blas/types.zig` to the repository layout and mention the new operations in the module map.

- [ ] **Step 2: Update `README.md`**

Add `gemv`, `ger`, `gemm` to the Phase 1/2 capability list and mention `examples/matrix_ops.zig`.

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md README.md
git commit -m "docs: document Phase 2 BLAS Level-2/3 additions"
```

---

## Self-Review Checklist

After completing all tasks, run:

```bash
zig build test
zig build example
```

Both must pass. Verify:
- [ ] `src/blas/types.zig` exists and exports `Transpose`.
- [ ] `src/blas.zig` exports `gemv`, `ger`, `gemm`.
- [ ] No TODO/TBD placeholders remain in source files.
- [ ] All new files are committed to git.
