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

## Build

Requires Zig 0.17.0-dev or later.

```sh
zig build test           # run unit tests
zig build example        # run matrix_ops demo
zig build example-vector # run vector_ops demo
```

## Quick Examples

- `examples/vector_ops.zig` — BLAS Level-1 vector operations
- `examples/matrix_ops.zig` — BLAS Level-2/3 matrix operations
