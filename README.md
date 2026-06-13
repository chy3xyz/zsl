# zsl — Zig Scientific Library

A pure-Zig scientific computing library. `zsl` ports the core primitives of the
[V Scientific Library (VSL)](https://github.com/vlang/vsl) to idiomatic Zig,
with explicit allocator passing, compile-time type safety, and no required
external dependencies.

## Status

The library is under active development. The following areas are already
usable and covered by inline tests:

- **Linear algebra**: dense `Vector(T)` / `Matrix(T)`, BLAS Level-1/2/3,
  LU factorization, linear solvers, symmetric eigenvalue decomposition
- **Statistics & metrics**: column statistics, classification/regression metrics
- **Numerical utilities**: root finders, polynomials, primes, histograms,
  combinatorial iterators
- **Preprocessing & model selection**: scalers, encoders, binning,
  train-test split, k-fold split
- **Geometry & graphs**: quaternions, Perlin/Simplex noise, spatial bins,
  shortest-path algorithms
- **Special functions**: gamma, digamma, error functions, Bessel functions,
  interpolation
- **Signal processing**: radix-2 FFT/IFFT
- **Machine learning**: `Data(T)` container, K-Means, linear/KNN/logistic
  regression, SVM, decision trees, LASSO, random forests
- **Easing & plotting**: easing/tweening functions, Plotly HTML plotting
- **I/O**: CSV reader/writer

## Requirements

- [Zig](https://ziglang.org) `0.17.0-dev.813+2153f8143` or later

## Quick start

Add `zsl` as a dependency in your `build.zig.zon`:

```zon
.{
    .name = .myproject,
    .version = "0.1.0",
    .dependencies = .{
        .zsl = .{
            .url = "https://github.com/chy3xyz/zsl/archive/refs/heads/master.tar.gz",
        },
    },
}
```

Then import it in your `build.zig`:

```zig
const zsl = b.dependency("zsl", .{});
exe.root_module.addImport("zsl", zsl.module("zsl"));
```

Use it in your code:

```zig
const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const M = zsl.la.Matrix(f64);

    var a = try M.fromRowSlice(allocator, 2, 2, &[_]f64{
        1.0, 2.0,
        3.0, 4.0,
    });
    defer a.deinit(allocator);

    const det = try zsl.la.matrix_ops.det(allocator, a);
    std.debug.print("det = {d}\n", .{det});
}
```

## Building and testing

```sh
zig build test                  # run the full test suite
zig build example               # run the matrix_ops demo
zig build example-vector        # run the vector_ops demo
zig build example-blas          # run the blas_core demo
zig build example-lapack        # run the lapack_solve demo
zig build example-eigen         # run the eigen_ops demo
zig build example-statistics    # run the statistics_ops demo
zig build example-metrics       # run the metrics_ops demo
zig build example-roots         # run the roots_ops demo
zig build example-poly          # run the poly_ops demo
zig build example-prime         # run the prime_ops demo
zig build example-dist          # run the dist_ops demo
zig build example-iter          # run the iter_ops demo
zig build example-preprocessing # run the preprocessing_ops demo
zig build example-model-selection  # run the model_selection_ops demo
zig build example-quaternion    # run the quaternion_ops demo
zig build example-noise         # run the noise_ops demo
zig build example-consts        # run the consts_ops demo
zig build example-fun           # run the fun_ops demo
zig build example-graph         # run the graph_ops demo
zig build example-gm            # run the gm_ops demo
zig build example-kmeans        # run the kmeans_ops demo
zig build example-linreg        # run the linreg_ops demo
zig build example-knn           # run the knn_ops demo
zig build example-deriv         # run the deriv_ops demo
zig build example-diff          # run the diff_ops demo
zig build example-fft           # run the fft_ops demo
zig build example-easings       # run the easings_ops demo
zig build example-csv           # run the csv_ops demo
zig build example-ml-advanced   # run the ml_advanced_ops demo
zig build example-plot          # run the plot_ops demo
```

Run `zig build --help` to see all available steps.

## Examples

The `examples/` directory contains small, self-contained demos for most modules:

- `examples/vector_ops.zig` — BLAS Level-1 vector operations
- `examples/matrix_ops.zig` — dense matrix determinant / inverse / solve
- `examples/blas_core.zig` — symmetric and triangular BLAS operations
- `examples/lapack_solve.zig` — LU-based linear system solver
- `examples/eigen_ops.zig` — symmetric matrix eigenvalue decomposition
- `examples/statistics_ops.zig` — column statistics and correlation/covariance
- `examples/metrics_ops.zig` — classification and regression metrics
- `examples/roots_ops.zig` — root finding
- `examples/poly_ops.zig` — polynomial operations
- `examples/prime_ops.zig` — prime utilities
- `examples/dist_ops.zig` — histograms
- `examples/iter_ops.zig` — combinatorial iterators
- `examples/preprocessing_ops.zig` — scalers, encoders, binning
- `examples/model_selection_ops.zig` — train-test split and k-fold
- `examples/quaternion_ops.zig` — quaternion constructors, algebra, rotation
- `examples/noise_ops.zig` — Perlin and Simplex noise
- `examples/consts_ops.zig` — physical and numeric constants
- `examples/fun_ops.zig` — special functions and interpolation
- `examples/graph_ops.zig` — graph shortest paths
- `examples/gm_ops.zig` — spatial binning / neighbor search
- `examples/kmeans_ops.zig` — K-Means clustering
- `examples/linreg_ops.zig` — linear regression
- `examples/knn_ops.zig` — K-Nearest Neighbors
- `examples/deriv_ops.zig` — numerical derivatives
- `examples/diff_ops.zig` — automatic differentiation step-size
- `examples/fft_ops.zig` — FFT and spectrum analysis
- `examples/easings_ops.zig` — easing function samples
- `examples/csv_ops.zig` — CSV read/write
- `examples/ml_advanced_ops.zig` — logistic regression and other advanced ML demos
- `examples/plot_ops.zig` — Plotly HTML plotting

## Project layout

```text
src/
├── root.zig          # public re-exports
├── errors.zig        # shared error set
├── util.zig          # numeric type constraints
├── float.zig         # floating-point helpers
├── la.zig            # Vector / Matrix types
├── la/               # matrix ops, Jacobi, statistics
├── blas.zig          # BLAS Level-1/2/3
├── lapack.zig        # LAPACK re-exports
├── metrics.zig       # classification / regression metrics
├── roots.zig         # root finders
├── poly.zig          # polynomials
├── prime.zig         # prime utilities
├── dist.zig          # histograms
├── iter.zig          # combinatorial iterators
├── preprocessing.zig # scalers / encoders / binning
├── model_selection.zig
├── quaternion.zig    # quaternion math
├── noise.zig         # Perlin / Simplex noise
├── consts.zig        # physical / numeric constants
├── fun.zig           # special functions
├── graph.zig         # graph algorithms
├── gm.zig            # geometry bins
├── ml.zig            # machine learning re-exports
├── ml/               # data, kmeans, linreg, knn, logreg, svm, trees, etc.
├── deriv.zig         # numerical derivatives
├── diff.zig          # step-size selection
├── fft.zig           # FFT
├── easings.zig       # easing functions
├── inout.zig         # I/O re-exports
├── inout/csv.zig     # CSV reader/writer
└── plot.zig          # Plotly HTML plotting

examples/             # runnable demos
build.zig             # build configuration
build.zig.zon         # package manifest
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, style notes, and
the pull-request workflow.

## License

`zsl` is released under the [MIT License](LICENSE).

The original [VSL](https://github.com/vlang/vsl) reference implementation is
also MIT licensed. A local reference copy may be kept under `_ref/vsl/` during
development but is not part of the published package.
