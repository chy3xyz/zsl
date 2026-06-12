# zsl BLAS Level-2/3 — Design Spec

> Date: 2026-06-12
> Status: Approved
> Scope: Phase 2 of the VSL → Zig port

## 1. Goal

Extend the zsl linear-algebra layer with BLAS Level-2 (matrix-vector) and Level-3
(matrix-matrix) operations. This phase builds directly on `Vector(T)`, `Matrix(T)`,
and the shared `Error` set introduced in Phase 1.

## 2. Decisions

| Topic | Decision |
|-------|----------|
| Operations | `gemv`, `ger`, `gemm`. Symmetric/triangular ops deferred. |
| Transpose support | `Transpose` enum: `.no_trans`, `.trans`, `.conj_trans`. `conj_trans` treated identically to `.trans` for real types. |
| Numeric precision | Generic over `f16`, `f32`, `f64`, `f128`; primary tests target `f32` and `f64`. |
| Implementation | Pure Zig, zero external dependencies. |
| Parallelism | Deferred; `gemm` uses serial blocking only. |
| Blocking | Simple row/column blocking with a comptime block size (default 32). |

## 3. Project Layout

```text
zsl/
├── src/
│   ├── blas.zig           # extended with Level-2/3 ops
│   └── blas/
│       └── types.zig      # Transpose enum
└── examples/
    └── matrix_ops.zig     # gemv / gemm demo
```

## 4. Module Responsibilities

### 4.1 `src/blas/types.zig`

```zig
pub const Transpose = enum {
    no_trans,
    trans,
    conj_trans,
};
```

For real floating-point types `.conj_trans` is semantically identical to `.trans`.

### 4.2 `src/blas.zig`

**`gemv`**

```zig
pub fn gemv(
    comptime T: type,
    trans_a: Transpose,
    alpha: T,
    a: Matrix(T),
    x: Vector(T),
    beta: T,
    y: *Vector(T),
) Error!void
```

Computes `y ← alpha·op(A)·x + beta·y`.

- If `trans_a == .no_trans`, `op(A)` is `A` and shapes must satisfy `A.rows == y.len`, `A.cols == x.len`.
- If `trans_a != .no_trans`, `op(A)` is `Aᵀ` and shapes must satisfy `A.cols == y.len`, `A.rows == x.len`.
- `y` must already be allocated with the correct length.
- `beta == 0` means `y` is overwritten (treated as zero before scaling).

**`ger`**

```zig
pub fn ger(
    comptime T: type,
    alpha: T,
    x: Vector(T),
    y: Vector(T),
    a: *Matrix(T),
) Error!void
```

Computes `A ← alpha·x·yᵀ + A`.

- `x.len` must equal `A.rows`.
- `y.len` must equal `A.cols`.
- `A` must already be allocated.

**`gemm`**

```zig
pub fn gemm(
    comptime T: type,
    trans_a: Transpose,
    trans_b: Transpose,
    alpha: T,
    a: Matrix(T),
    b: Matrix(T),
    beta: T,
    c: *Matrix(T),
) Error!void
```

Computes `C ← alpha·op(A)·op(B) + beta·C`.

- Inner dimensions of `op(A)` and `op(B)` must match.
- Output dimensions of `C` must match the outer dimensions.
- `C` must already be allocated.
- `gemm` uses a blocked algorithm when both operands are `.no_trans` and dimensions are
  larger than the block size; otherwise it falls back to a simple triple loop.

## 5. API Conventions

- Types: `PascalCase` (`Transpose`).
- Functions: `snake_case`.
- `comptime T` is the first parameter for generic functions.
- Control enums (`Transpose`) are the next parameters.
- Scalars `alpha`/`beta` follow the enums.
- Input matrices/vectors follow scalars.
- Output pointer (`*Vector(T)` or `*Matrix(T)`) is last.
- All functions validate dimensions and return `ShapeMismatch` on mismatch.

## 6. Error Handling

- `ShapeMismatch` — operand dimensions are incompatible.
- `InvalidDimension` — zero-sized matrices or vectors where non-zero is required.
- `NotImplemented` — reserved for future unsupported transpose combinations or operations.

## 7. Testing Strategy

- Inline `test {}` blocks in `src/blas.zig`.
- Numerical assertions use `float.approxEqAbs` / `approxEqRel`.
- Coverage targets:
  - `gemv` with `.no_trans` and `.trans` for `f32` and `f64`.
  - `ger` rank-one update for `f32` and `f64`.
  - `gemm` with all four transpose combinations.
  - `gemm` with `beta = 0`, `beta = 1`, and `beta = 0.5`.
  - Shape-mismatch error cases for all three operations.

## 8. Build / Tooling

- No new build steps. Existing `zig build test` and `zig build example` continue to work.
- `examples/matrix_ops.zig` is added to the example step automatically via `build.zig`.

## 9. Out of Scope

- Symmetric (`symv`, `syrk`) and triangular (`trmv`, `trsv`, `trmm`, `trsm`) operations.
- Banded and packed matrix formats.
- Parallel / threaded `gemm`.
- C BLAS backend.
- Complex number support.

## 10. Success Criteria

- `zig build test` passes with no failures.
- `zig build example` runs `matrix_ops` and prints expected results.
- All new functions have passing numerical tests for `f32` and `f64`.
