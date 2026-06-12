# zsl LAPACK Linear Solvers — Design Spec

> Date: 2026-06-12
> Status: Approved
> Scope: Phase 4 of the VSL → Zig port

## 1. Goal

Add a `lapack` module to zsl that mirrors VSL's `lapack64` linear-solver
foundation: LU factorization, forward/backward substitution, and a combined
linear-system solver.

## 2. Decisions

| Topic | Decision |
|-------|----------|
| Routines | `dgetf2`, `dgetrf`, `dlaswp`, `dgetrs`, `dgesv`. `dgetri` deferred. |
| Numeric type | Generic `comptime T` for consistency with `zsl.blas`; primary tests target `f64`. |
| Pivot storage | `[]usize` zero-based pivot indices, matching VSL's convention. |
| In-place | `dgetrf` stores `L` (unit diagonal) and `U` in the input matrix. |
| Implementation | Pure Zig, zero dependencies, reuses `zsl.blas` primitives (`iamax`, `swap`, `scal`, `ger`). |
| Blocking | `dgetrf` delegates to `dgetf2` (unblocked) for simplicity; mirrors VSL's current default. |

## 3. Project Layout

```text
zsl/
├── src/
│   ├── root.zig            # Re-export zsl.lapack
│   ├── lapack.zig          # Public lapack module
│   └── lapack/
│       └── lu.zig          # dgetf2, dgetrf, dlaswp, dgetrs, dgesv
└── examples/
    └── lapack_solve.zig    # dgesv demo
```

## 4. API

```zig
pub fn dgetf2(
    comptime T: type,
    m: usize,
    n: usize,
    a: *Matrix(T),
    ipiv: []usize,
) Error!bool

pub fn dgetrf(
    comptime T: type,
    m: usize,
    n: usize,
    a: *Matrix(T),
    ipiv: []usize,
) Error!bool

pub fn dlaswp(
    comptime T: type,
    a: *Matrix(T),
    k1: usize,
    k2: usize,
    ipiv: []const usize,
    incx: isize,
) Error!void

pub fn dgetrs(
    comptime T: type,
    trans_a: Transpose,
    a: Matrix(T),
    ipiv: []const usize,
    b: *Matrix(T),
) Error!void

pub fn dgesv(
    comptime T: type,
    a: *Matrix(T),
    ipiv: []usize,
    b: *Matrix(T),
) Error!bool
```

- `dgetf2`/`dgetrf` return `true` if nonsingular, `false` if a zero pivot was
  encountered. The factorization is still performed on singular input.
- `dlaswp` applies a sequence of row interchanges to `a`.
- `dgetrs` solves `A·X = B` or `Aᵀ·X = B` using the LU factorization in `a`.
- `dgesv` factors `A` and solves `A·X = B` in one call.

## 5. API Conventions

- `comptime T` is the first parameter.
- `m`/`n` are explicit dimensions (allowing submatrix views).
- Pivot indices are zero-based.
- Errors: `ShapeMismatch` for dimension mismatches; `InvalidDimension` for zero
  dimensions where not allowed.

## 6. Error Handling

- `ShapeMismatch` — matrix/vector dimensions are incompatible.
- `InvalidDimension` — zero-sized operands where non-zero is required.
- `NotImplemented` — reserved for unsupported transpose modes.

## 7. Testing Strategy

- Factor a known matrix, reconstruct `P·L·U`, compare with original.
- Solve a system with a known solution and verify the residual.
- Test rectangular `m > n` and `m < n` factorizations.
- Test singular matrix detection.

## 8. Build / Tooling

- No new build steps.
- `examples/lapack_solve.zig` added as `example-lapack` step.

## 9. Out of Scope

- `dgetri` (matrix inversion), `dpotrf` (Cholesky), QR, SVD, eigenvalue routines.
- Blocked `dgetrf` with Level-3 updates.
- LAPACKE C backend.

## 10. Success Criteria

- `zig build test` passes.
- `zig build example-lapack` runs the demo and prints expected results.
- LU factorization reconstructs the original matrix within tolerance.
- Linear solve recovers a known solution within tolerance.
