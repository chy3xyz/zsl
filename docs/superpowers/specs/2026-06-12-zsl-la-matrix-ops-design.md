.# zsl.la Matrix Operations — Design Spec

> Date: 2026-06-12
> Status: Approved
> Scope: Phase 5 of the VSL → Zig port

## 1. Goal

Add high-level dense matrix operations to `zsl.la`, mirroring VSL's
`la/matrix_ops.v` and `la/densesol.v`. These operations wrap the underlying
BLAS/LAPACK primitives into a convenient API.

## 2. Decisions

| Topic | Decision |
|-------|----------|
| Module | `src/la/matrix_ops.zig` re-exported by `src/la.zig`. |
| Operations | `det`, `inverse_small`, `inverse`, `solve`. |
| Numeric type | Generic `comptime T`; tests target `f64`. |
| Implementation | Pure Zig, reuses `zsl.lapack.lu`. |
| Error handling | Returns `error.SingularMatrix` when appropriate instead of panicking. |

## 3. Project Layout

```text
zsl/
├── src/
│   ├── la.zig              # Re-export matrix_ops
│   └── la/
│       └── matrix_ops.zig  # det, inverse, solve
└── examples/
    └── la_ops.zig          # det / inv / solve demo
```

## 4. API

```zig
pub fn det(comptime T: type, a: Matrix(T)) Error!T

pub fn inverse_small(comptime T: type, a: Matrix(T), out: *Matrix(T), tol: T) Error!T

pub fn inverse(comptime T: type, a: Matrix(T), allocator: std.mem.Allocator) Error!Matrix(T)

pub fn solve(
    comptime T: type,
    a: Matrix(T),
    b: Vector(T),
    x: *Vector(T),
    allocator: std.mem.Allocator,
) Error!void
```

- `det` computes the determinant from the LU factorization.
- `inverse_small` computes the inverse of 1×1, 2×2, or 3×3 matrices and returns
  the determinant.
- `inverse` computes the inverse of a general square matrix using `dgetrf` and
  `dgetri` (or repeated `dgetrs` with identity columns).
- `solve` solves `A·x = b` for a single right-hand side.

## 5. API Conventions

- `comptime T` first.
- Input matrices/vectors follow.
- Output pointer or allocator follows inputs.
- `tol` parameter for small-matrix inversion.

## 6. Error Handling

- `ShapeMismatch` — non-square matrix for `det`/`inverse`, mismatched dimensions
  for `solve`.
- `SingularMatrix` — zero determinant or zero pivot.

## 7. Testing Strategy

- `det` on known 2×2 and 3×3 matrices.
- `inverse_small` round-trip: `A * A^-1 == I`.
- `inverse` round-trip for larger matrices.
- `solve` recovers a known solution.

## 8. Build / Tooling

- No new build steps.
- `examples/la_ops.zig` added as `example-la` step.

## 9. Out of Scope

- SVD, eigenvalue, Cholesky wrappers.
- Pseudo-inverse for non-square matrices.
- Parallel implementations.

## 10. Success Criteria

- `zig build test` passes.
- `zig build example-la` runs and prints expected results.
- Round-trip tests pass within tolerance.
