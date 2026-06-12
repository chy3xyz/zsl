# zsl Core BLAS Level-2/3 — Design Spec

> Date: 2026-06-12
> Status: Approved
> Scope: Phase 3 of the VSL → Zig port

## 1. Goal

Complete the core subset of BLAS Level-2 and Level-3 routines so that zsl's
`blas` module closely mirrors VSL's `blas64` capabilities for dense matrices.

## 2. Decisions

| Topic | Decision |
|-------|----------|
| Level-2 ops | `symv`, `syr`, `syr2`, `trmv`, `trsv`. Banded/packed variants deferred. |
| Level-3 ops | `syrk`, `syr2k`, `trmm`, `trsm`. Banded/packed variants deferred. |
| Control enums | `Transpose` (existing), `Uplo` (upper/lower), `Side` (left/right), `Diagonal` (unit/non_unit). |
| Storage | Dense `Matrix(T)` only; symmetric/triangular matrices store full matrix and access only the requested triangle. |
| Numeric precision | Generic over `f16`, `f32`, `f64`, `f128`; primary tests target `f32` and `f64`. |
| Implementation | Pure Zig, zero external dependencies. |
| Performance | Correctness first; no parallelism, optional simple blocking for `syrk`. |

## 3. Project Layout

```text
zsl/
├── src/
│   ├── blas/
│   │   └── types.zig      # Transpose, Uplo, Side, Diagonal
│   └── blas.zig           # Core Level-2/3 routines and tests
└── examples/
    └── blas_core.zig      # symv / trsv / syrk / trsm demo
```

## 4. Module Responsibilities

### 4.1 `src/blas/types.zig`

```zig
pub const Transpose = enum { no_trans, trans, conj_trans };
pub const Uplo = enum { upper, lower };
pub const Side = enum { left, right };
pub const Diagonal = enum { unit, non_unit };
```

### 4.2 `src/blas.zig`

All functions accept `comptime T: type` as the first parameter and validate that
`T` is a floating-point type via `util.Float(T)`.

**Level-2**

```zig
pub fn symv(
    comptime T: type,
    uplo: Uplo,
    alpha: T,
    a: Matrix(T),
    x: Vector(T),
    beta: T,
    y: *Vector(T),
) Error!void
```

Computes `y ← alpha·A·x + beta·y` where `A` is symmetric.

```zig
pub fn syr(
    comptime T: type,
    uplo: Uplo,
    alpha: T,
    x: Vector(T),
    a: *Matrix(T),
) Error!void
```

Computes `A ← alpha·x·xᵀ + A` where `A` is symmetric.

```zig
pub fn syr2(
    comptime T: type,
    uplo: Uplo,
    alpha: T,
    x: Vector(T),
    y: Vector(T),
    a: *Matrix(T),
) Error!void
```

Computes `A ← alpha·x·yᵀ + alpha·y·xᵀ + A` where `A` is symmetric.

```zig
pub fn trmv(
    comptime T: type,
    uplo: Uplo,
    trans_a: Transpose,
    diag: Diagonal,
    a: Matrix(T),
    x: *Vector(T),
) Error!void
```

Computes `x ← op(A)·x` where `A` is triangular.

```zig
pub fn trsv(
    comptime T: type,
    uplo: Uplo,
    trans_a: Transpose,
    diag: Diagonal,
    a: Matrix(T),
    x: *Vector(T),
) Error!void
```

Solves `op(A)·x = b` where `A` is triangular; `x` initially contains `b` and is
overwritten with the solution.

**Level-3**

```zig
pub fn syrk(
    comptime T: type,
    uplo: Uplo,
    trans_a: Transpose,
    alpha: T,
    a: Matrix(T),
    beta: T,
    c: *Matrix(T),
) Error!void
```

Computes `C ← alpha·op(A)·op(A)ᵀ + beta·C` where `C` is symmetric.

```zig
pub fn syr2k(
    comptime T: type,
    uplo: Uplo,
    trans_a: Transpose,
    alpha: T,
    a: Matrix(T),
    b: Matrix(T),
    beta: T,
    c: *Matrix(T),
) Error!void
```

Computes `C ← alpha·op(A)·op(B)ᵀ + alpha·op(B)·op(A)ᵀ + beta·C` where `C` is
symmetric.

```zig
pub fn trmm(
    comptime T: type,
    side: Side,
    uplo: Uplo,
    trans_a: Transpose,
    diag: Diagonal,
    alpha: T,
    a: Matrix(T),
    b: *Matrix(T),
) Error!void
```

Computes `B ← alpha·op(A)·B` (left) or `B ← alpha·B·op(A)` (right) where `A` is
triangular.

```zig
pub fn trsm(
    comptime T: type,
    side: Side,
    uplo: Uplo,
    trans_a: Transpose,
    diag: Diagonal,
    alpha: T,
    a: Matrix(T),
    b: *Matrix(T),
) Error!void
```

Solves `op(A)·X = alpha·B` (left) or `X·op(A) = alpha·B` (right) where `A` is
triangular; `B` initially contains the right-hand side and is overwritten with
`X`.

## 5. API Conventions

- Types: `PascalCase`.
- Functions: `snake_case`.
- `comptime T` first, then control enums, then scalars, then inputs, then
  output pointer.
- All functions validate dimensions and return `ShapeMismatch` on mismatch.
- `Diagonal.unit` means the diagonal is implicitly 1 and is not read from `A`.

## 6. Error Handling

- `ShapeMismatch` — operand dimensions are incompatible.
- `InvalidDimension` — zero-sized matrices or vectors where non-zero is required.
- `NotImplemented` — reserved for unsupported combinations.

## 7. Testing Strategy

- Inline `test {}` blocks in `src/blas.zig`.
- Numerical assertions use `float.approxEqAbs` / `approxEqRel`.
- Coverage targets for `f32` and `f64`:
  - `symv` with `.upper` and `.lower`.
  - `syr` and `syr2` update only the requested triangle.
  - `trmv` / `trsv` with `.unit` and `.non_unit`, `.no_trans` and `.trans`.
  - `syrk` with `.no_trans` and `.trans`.
  - `syr2k` basic case.
  - `trmm` / `trsm` with `.left` and `.right`.
  - Shape-mismatch error cases.

## 8. Build / Tooling

- No new build steps.
- `examples/blas_core.zig` added to `build.zig` as `example-blas` step.

## 9. Out of Scope

- Banded matrix routines (`gbmv`, `sbmv`, `tbmv`, `tbsv`, etc.).
- Packed matrix routines (`spmv`, `tpmv`, `tpsv`, etc.).
- Hermitian routines (`hemv`, `herk`, `her2k`, `hemm`) — real-only for now.
- Parallel / threaded kernels.
- C BLAS backend.

## 10. Success Criteria

- `zig build test` passes with no failures.
- `zig build example-blas` runs `blas_core` and prints expected results.
- All new routines have passing numerical tests for `f32` and `f64`.
