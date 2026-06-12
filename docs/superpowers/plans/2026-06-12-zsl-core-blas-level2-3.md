# zsl Core BLAS Level-2/3 Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the core subset of BLAS Level-2/3 operations in `src/blas.zig`, closely aligned with VSL's `blas64` API.

**Architecture:** Pure-Zig generic functions over floating-point types, operating on `Vector(T)` and `Matrix(T)`. Control enums live in `src/blas/types.zig`.

**Tech Stack:** Zig 0.17.0-dev, `std.Build`, `std.testing.allocator`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/blas/types.zig` | `Transpose`, `Uplo`, `Side`, `Diagonal` enums. |
| `src/blas.zig` | Core Level-2/3 routines and tests. |
| `examples/blas_core.zig` | Demo of `symv`, `trsv`, `syrk`, `trsm`. |
| `build.zig` | Add `example-blas` step. |
| `AGENTS.md`, `README.md` | Document new routines. |

---

## Task 1: Extend `src/blas/types.zig`

**Files:**
- Modify: `src/blas/types.zig`

- [ ] **Step 1: Add enums**

```zig
pub const Uplo = enum { upper, lower };
pub const Side = enum { left, right };
pub const Diagonal = enum { unit, non_unit };
```

- [ ] **Step 2: Add tests**

```zig
test "BLAS control enums" {
    const u: Uplo = .upper;
    const s: Side = .left;
    const d: Diagonal = .unit;
    try std.testing.expect(u == .upper);
    try std.testing.expect(s == .left);
    try std.testing.expect(d == .unit);
}
```

- [ ] **Step 3: Run `zig build test`**

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/blas/types.zig
git commit -m "feat(blas): add Uplo, Side, Diagonal enums"
```

---

## Task 2: Implement Level-2 routines

**Files:**
- Modify: `src/blas.zig`

### 2.1 `symv`

- [ ] **Step 1: Add stub returning `ShapeMismatch`**
- [ ] **Step 2: Add failing tests** for `.upper` and `.lower`, `f32` and `f64`
- [ ] **Step 3: Run `zig build test`** (expected FAIL)
- [ ] **Step 4: Implement `symv`** — only access the requested triangle
- [ ] **Step 5: Run `zig build test`** (expected PASS)
- [ ] **Step 6: Add shape-mismatch test**

### 2.2 `syr`

- [ ] **Step 1: Add stub**
- [ ] **Step 2: Add failing test** — verify only requested triangle is updated
- [ ] **Step 3: Run `zig build test`** (expected FAIL)
- [ ] **Step 4: Implement `syr`**
- [ ] **Step 5: Run `zig build test`** (expected PASS)

### 2.3 `syr2`

- [ ] **Step 1: Add stub**
- [ ] **Step 2: Add failing test**
- [ ] **Step 3: Run `zig build test`** (expected FAIL)
- [ ] **Step 4: Implement `syr2`**
- [ ] **Step 5: Run `zig build test`** (expected PASS)

### 2.4 `trmv`

- [ ] **Step 1: Add stub**
- [ ] **Step 2: Add failing tests** for `.upper`/`.lower`, `.unit`/`.non_unit`, `.no_trans`/`.trans`
- [ ] **Step 3: Run `zig build test`** (expected FAIL)
- [ ] **Step 4: Implement `trmv`** — in-place triangular matrix-vector multiply
- [ ] **Step 5: Run `zig build test`** (expected PASS)

### 2.5 `trsv`

- [ ] **Step 1: Add stub**
- [ ] **Step 2: Add failing tests** for forward/backward substitution
- [ ] **Step 3: Run `zig build test`** (expected FAIL)
- [ ] **Step 4: Implement `trsv`** — in-place triangular solve
- [ ] **Step 5: Run `zig build test`** (expected PASS)

- [ ] **Step 6: Commit Level-2**

```bash
git add src/blas.zig
git commit -m "feat(blas): add Level-2 core routines (symv, syr, syr2, trmv, trsv)"
```

---

## Task 3: Implement Level-3 routines

**Files:**
- Modify: `src/blas.zig`

### 3.1 `syrk`

- [ ] **Step 1: Add stub**
- [ ] **Step 2: Add failing tests** for `.no_trans` and `.trans`, `.upper`/`.lower`
- [ ] **Step 3: Run `zig build test`** (expected FAIL)
- [ ] **Step 4: Implement `syrk`** — only update requested triangle
- [ ] **Step 5: Run `zig build test`** (expected PASS)

### 3.2 `syr2k`

- [ ] **Step 1: Add stub**
- [ ] **Step 2: Add failing test**
- [ ] **Step 3: Run `zig build test`** (expected FAIL)
- [ ] **Step 4: Implement `syr2k`**
- [ ] **Step 5: Run `zig build test`** (expected PASS)

### 3.3 `trmm`

- [ ] **Step 1: Add stub**
- [ ] **Step 2: Add failing tests** for `.left`/`.right`, `.upper`/`.lower`, `.unit`/`.non_unit`
- [ ] **Step 3: Run `zig build test`** (expected FAIL)
- [ ] **Step 4: Implement `trmm`** — triangular matrix-matrix multiply
- [ ] **Step 5: Run `zig build test`** (expected PASS)

### 3.4 `trsm`

- [ ] **Step 1: Add stub**
- [ ] **Step 2: Add failing tests** for `.left`/`.right`, `.upper`/`.lower`, `.unit`/`.non_unit`
- [ ] **Step 3: Run `zig build test`** (expected FAIL)
- [ ] **Step 4: Implement `trsm`** — triangular solve with multiple right-hand sides
- [ ] **Step 5: Run `zig build test`** (expected PASS)

- [ ] **Step 6: Commit Level-3**

```bash
git add src/blas.zig
git commit -m "feat(blas): add Level-3 core routines (syrk, syr2k, trmm, trsm)"
```

---

## Task 4: Add `examples/blas_core.zig`

**Files:**
- Create: `examples/blas_core.zig`
- Modify: `build.zig`

- [ ] **Step 1: Write example**

Demonstrate `symv`, `trsv`, `syrk`, `trsm` on small `f64` matrices.

- [ ] **Step 2: Add `example-blas` step to `build.zig`**

```zig
const blas_example_step = b.step("example-blas", "Run blas_core example");
const blas_exe = b.addExecutable(.{
    .name = "blas_core",
    .root_module = b.createModule(.{
        .root_source_file = b.path("examples/blas_core.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zsl", .module = zsl_mod }},
    }),
});
blas_example_step.dependOn(&b.addRunArtifact(blas_exe).step);
```

- [ ] **Step 3: Run `zig build example-blas`**

Expected: PASS and print results.

- [ ] **Step 4: Commit**

```bash
git add examples/blas_core.zig build.zig
git commit -m "feat(examples): add blas_core demo"
```

---

## Task 5: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Update `README.md`**

Add Phase 3 routines to the capability list and mention `zig build example-blas`.

- [ ] **Step 2: Update `AGENTS.md`**

Add `examples/blas_core.zig` to the repository layout and command list.

- [ ] **Step 3: Commit**

```bash
git add README.md AGENTS.md
git commit -m "docs: document Phase 3 core BLAS Level-2/3 routines"
```

---

## Task 6: Final verification

- [ ] **Run `zig build test`**

Expected: PASS.

- [ ] **Run `zig build example-blas`**

Expected: PASS.

- [ ] **Leak scan**

Verify all allocations in `la.zig` still have matching `errdefer`/`defer`.

- [ ] **Commit if any fixes**

---

## Self-Review Checklist

After completing all tasks:

- [ ] `src/blas/types.zig` exports `Transpose`, `Uplo`, `Side`, `Diagonal`.
- [ ] `src/blas.zig` exports `symv`, `syr`, `syr2`, `trmv`, `trsv`, `syrk`, `syr2k`, `trmm`, `trsm`.
- [ ] No TODO/TBD placeholders remain in source files.
- [ ] All new files are committed to git.
- [ ] `zig build test` passes.
- [ ] `zig build example-blas` passes.
