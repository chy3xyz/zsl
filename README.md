# zsl — Zig Scientific Library

A pure-Zig scientific computing library. This is a Zig port of the scientific-computing primitives found in [VSL](https://github.com/vlang/vsl), redesigned for idiomatic Zig.

## Status

Phase 1 implements the core linear-algebra foundation:

- `util` — numeric type constraints and helpers
- `float` — epsilon, approximate equality, finite checks
- `errors` — shared error set
- `la` — dense `Vector(T)` and `Matrix(T)` containers
- `blas` — BLAS Level-1 operations

## Build

Requires Zig 0.17.0-dev or later.

```sh
zig build test      # run unit tests
zig build example   # run vector_ops demo
```

## Quick Example

See `examples/vector_ops.zig`.
