# zsl — Zig Scientific Library

A pure-Zig scientific computing library. This is a Zig port of the scientific-computing primitives found in [VSL](https://github.com/vlang/vsl), redesigned for idiomatic Zig.

## Status

Phase 1 implements the core linear-algebra foundation:

- `util` — numeric type constraints and helpers
- `float` — epsilon, approximate equality, finite checks
- `errors` — shared error set
- `la` — dense `Vector(T)` and `Matrix(T)` containers
- `blas` — BLAS Level-1 operations

Phase 2 extends BLAS coverage to Level-2 and Level-3:

- `gemv` — matrix-vector multiply with transpose
- `ger` — rank-one update
- `gemm` — matrix-matrix multiply with transpose and blocking

Phase 3 completes the core BLAS Level-2/3 subset:

- `symv`, `syr`, `syr2` — symmetric matrix-vector / rank updates
- `trmv`, `trsv` — triangular matrix-vector / solve
- `syrk`, `syr2k` — symmetric rank-k / 2k updates
- `trmm`, `trsm` — triangular matrix-matrix multiply / solve

Phase 4 starts the LAPACK layer with linear solvers:

- `dgetf2` / `dgetrf` — LU factorization (unblocked)
- `dlaswp` — apply row permutations
- `dgetrs` — solve using LU factorization
- `dgesv` — factor and solve in one call

## Build

Requires Zig 0.17.0-dev or later.

```sh
zig build test           # run unit tests
zig build example        # run matrix_ops demo
zig build example-vector # run vector_ops demo
zig build example-blas   # run blas_core demo
zig build example-lapack # run lapack_solve demo
```

## Quick Examples

- `examples/vector_ops.zig` — BLAS Level-1 vector operations
- `examples/matrix_ops.zig` — BLAS Level-2/3 matrix operations
- `examples/blas_core.zig` — symmetric and triangular BLAS operations
- `examples/lapack_solve.zig` — LU-based linear system solver
