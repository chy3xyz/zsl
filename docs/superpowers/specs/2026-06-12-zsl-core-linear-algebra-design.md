# zsl Core Linear Algebra — Design Spec

> Date: 2026-06-12  
> Status: Approved  
> Scope: Phase 1 of the VSL → Zig port

## 1. Goal

Create the foundational Zig modules for a scientific computing library named **zsl** (Zig Scientific Library). This phase focuses on the core linear-algebra layer: utilities, floating-point helpers, a shared error set, dense `Vector`/`Matrix` types, and BLAS Level-1 operations. The reference VSL tree under `_ref/vsl/` provides the semantic baseline, but the Zig API is intentionally redesigned to be idiomatic Zig.

## 2. Decisions

| Topic | Decision |
|-------|----------|
| Modules ported first | Core linear algebra: `util`, `float`, `errors`, `la`, `blas`. |
| Implementation style | Pure Zig. No C BLAS/LAPACK dependencies in phase 1. |
| Numeric precision | Generic over floating-point types (`f16`, `f32`, `f64`, optionally `f128`) using Zig comptime constraints. |
| API style | Idiomatic Zig: `PascalCase` types, `snake_case` functions, explicit `Allocator`, 0-based `usize` indexing, no hidden globals. |
| Approach | Foundation-first vertical slice: build layers in dependency order, test each layer before moving up. |

## 3. Project Layout

```text
zsl/
├── build.zig              # zig build / zig build test
├── build.zig.zon          # package manifest
├── README.md              # quick start for users
├── AGENTS.md              # updated agent guide
├── src/
│   ├── root.zig           # public re-exports
│   ├── util.zig           # numeric type constraints + helpers
│   ├── float.zig          # epsilon, approximate equality, finite checks
│   ├── errors.zig         # shared Error error set
│   ├── la.zig             # Vector(T) and Matrix(T) dense types
│   └── blas.zig           # BLAS Level-1 operations on Vector(T)
└── examples/
    └── vector_ops.zig     # runnable demo: axpy / dot / nrm2 / scal
```

## 4. Module Responsibilities

### 4.1 `util.zig`

- `Float` comptime constraint: accepts `f16`, `f32`, `f64`, `f128`. Primary testing targets are `f32` and `f64`; `f16`/`f128` are supported where Zig's language/stdlib permits.
- `Real` alias/constraint for the scalar type backing a vector/matrix.
- Small index/size validation helpers shared by `la` and `blas`.

### 4.2 `float.zig`

- `eps(T: type) T` — machine epsilon for type `T`.
- `approxEqAbs(T, a, b, tol)` — absolute tolerance comparison.
- `approxEqRel(T, a, b, rel_tol, abs_tol)` — relative + absolute tolerance.
- `isFinite(T, x)` — true if `x` is finite.
- Helpers used by tests and by BLAS numerical checks.

### 4.3 `errors.zig`

A shared error set:

```zig
pub const Error = error{
    OutOfMemory,
    InvalidDimension,
    ShapeMismatch,
    IndexOutOfBounds,
    DivisionByZero,
    NotImplemented,
};
```

Module-specific errors can extend this via error-set union when necessary.

### 4.4 `la.zig`

Dense linear-algebra containers.

**`Vector(T)`**

```zig
pub fn Vector(comptime T: type) type {
    return struct {
        data: []T,
        len: usize,
        stride: usize,

        pub fn init(allocator: Allocator, len: usize) Error!@This() { ... }
        pub fn fromSlice(allocator: Allocator, slice: []const T) Error!@This() { ... }
        pub fn deinit(self: *@This(), allocator: Allocator) void { ... }

        pub fn get(self: @This(), i: usize) T { ... }
        pub fn set(self: @This(), i: usize, value: T) void { ... }
        pub fn rawData(self: @This()) []T { ... }
    };
}
```

- `stride` allows non-contiguous views (e.g., matrix rows/columns).
- `len` is the logical length; `data.len` may be larger when strided.
- `fromSlice` allocates a copy of the input slice and owns it.
- `rawData` returns the full backing storage (`data`), not a logical contiguous slice.

**`Matrix(T)`**

```zig
pub fn Matrix(comptime T: type) type {
    return struct {
        data: []T,
        rows: usize,
        cols: usize,
        row_stride: usize,
        col_stride: usize,

        pub fn init(allocator: Allocator, rows: usize, cols: usize) Error!@This() { ... }
        pub fn fromRowSlice(allocator: Allocator, rows: usize, cols: usize, slice: []const T) Error!@This() { ... }
        pub fn deinit(self: *@This(), allocator: Allocator) void { ... }

        pub fn get(self: @This(), r: usize, c: usize) T { ... }
        pub fn set(self: @This(), r: usize, c: usize, value: T) void { ... }
        pub fn row(self: @This(), r: usize) Vector(T) { ... }
        pub fn col(self: @This(), c: usize) Vector(T) { ... }
        pub fn transpose(self: @This()) @This() { ... }
    };
}
```

- Default storage is row-major: `row_stride = cols`, `col_stride = 1`.
- `fromRowSlice` copies the provided slice into owned storage.
- `transpose` returns a view by swapping `rows`/`cols` and `row_stride`/`col_stride`; no data is copied.
- `row` and `col` return strided `Vector(T)` views into the matrix data.

### 4.5 `blas.zig`

BLAS Level-1 operations on `Vector(T)`. All functions validate shape compatibility and return `ShapeMismatch` on mismatch.

Functions to implement in phase 1:

- `axpy(alpha, x, y)` — `y ← alpha*x + y`
- `dot(x, y)` — `x·y`
- `nrm2(x)` — `||x||₂`
- `scal(alpha, x)` — `x ← alpha*x`
- `copy(x, y)` — `y ← x` (`y` must already be allocated with matching length)
- `swap(x, y)` — swap contents of `x` and `y`
- `asum(x)` — `Σ |xᵢ|`
- `iamax(x)` — index of max absolute value

All Level-1 operations operate on existing `Vector(T)` instances; none allocate their own outputs in phase 1. Callers allocate input/output vectors via `Vector(T).init` or `fromSlice` before calling BLAS. Functions validate shape compatibility and return `ShapeMismatch` when lengths differ.

## 5. API Conventions

- Types: `PascalCase` (`Vector(T)`, `Matrix(T)`).
- Functions: `snake_case`.
- Allocator, when required, is the first parameter.
- `self` is used for methods where natural.
- Bounds checks are active in safe build modes.
- No global state; all state is explicit in parameters or receiver.

## 6. Error Handling

- Functions that can fail return `Error!T` or `Error!void`.
- Allocation failures surface as `OutOfMemory` from the standard allocator interface.
- Dimension/shape mismatches return `ShapeMismatch`.
- Invalid constructor arguments (e.g., `rows == 0` and `cols > 0`) return `InvalidDimension`.
- Out-of-range indexing returns `IndexOutOfBounds`.

## 7. Testing Strategy

- Tests live in inline `test {}` blocks next to the code they exercise.
- Use `std.testing.allocator` for leak detection in allocation tests.
- Numerical assertions use `float.approxEqAbs` / `approxEqRel` with small tolerances.
- `zig build test` runs all unit tests.
- `zig build example` builds and runs `examples/vector_ops.zig`.
- Initial coverage targets:
  - `util`: type-constraint acceptance/rejection.
  - `float`: epsilon and approximate equality for `f32`/`f64`.
  - `errors`: error-set membership.
  - `la`: init/deinit, indexing, slice round-trip, row/col views, transpose view, `f32` and `f64`.
  - `blas`: each Level-1 operation with known reference values for `f32` and `f64`, plus shape-mismatch error cases.

## 8. Build / Tooling

- `build.zig` declares:
  - a static library from `src/`,
  - a test step running `src/**/*.zig` tests,
  - an example step running `examples/vector_ops.zig`.
- `build.zig.zon` names the package `zsl`, version `0.1.0`.
- `AGENTS.md` is updated to describe the Zig project, build commands, and module map.
- No external dependencies in phase 1.

## 9. Out of Scope for Phase 1

- BLAS Level-2 / Level-3.
- LAPACK routines (solvers, eigenvalues, SVD).
- GPU backends (CUDA, Vulkan, OpenCL).
- Plotting, FFT, MPI, ML algorithms.
- C BLAS/LAPACK backend wrappers.
- Sparse matrices.
- Complex number support.

## 10. Success Criteria

- `zig build test` passes with no failures.
- `zig build example` runs and prints expected vector-operation results.
- `AGENTS.md` accurately describes the new Zig codebase.
- The public API in `src/root.zig` exposes `util`, `float`, `errors`, `la`, and `blas`.
