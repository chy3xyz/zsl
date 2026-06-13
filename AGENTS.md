# AGENTS.md — Project Guide for AI Coding Agents

> Last updated: 2026-06-12

## 1. Project overview

**zsl** (Zig Scientific Library) is a pure-Zig scientific computing library being
ported from the V-language VSL reference copy under `_ref/vsl/`. The codebase lives
at the workspace root and is built with `zig build`.

## 2. Technology stack

- **Primary language:** Zig (`*.zig`)
- **Build tool:** `zig build`
- **Package manifest:** `build.zig.zon`
- **Reference copy:** `_ref/vsl/` (VSL, read-only reference)

## 3. Repository layout

```text
zsl/
├── build.zig              # Build configuration
├── build.zig.zon          # Package manifest
├── README.md              # Human-facing project intro
├── AGENTS.md              # This file
├── src/
│   ├── root.zig           # Public re-exports
│   ├── util.zig           # Numeric type constraints
│   ├── float.zig          # Floating-point helpers
│   ├── errors.zig         # Shared error set
│   ├── la.zig             # Vector / Matrix types
│   ├── la/
│   │   ├── matrix_ops.zig # Determinant, inverse, solve
│   │   ├── jacobi.zig     # Symmetric eigenvalue decomposition
│   │   └── statistics.zig # Column statistics, correlation, covariance
│   ├── blas.zig           # BLAS Level-1/2/3 operations
│   ├── blas/
│   │   └── types.zig      # Transpose enum
│   ├── lapack.zig         # LAPACK re-exports
│   ├── lapack/
│   │   └── lu.zig         # LU factorization / linear solve
│   ├── metrics.zig        # Classification / regression metrics re-exports
│   ├── metrics/
│   │   ├── classification.zig # Binary classification metrics
│   │   └── regression.zig     # Regression metrics
│   ├── roots.zig          # Root finders
│   ├── poly.zig           # Polynomial operations
│   ├── prime.zig          # Prime utilities
│   ├── dist.zig           # Histograms
│   ├── iter.zig           # Combinatorial / range iterators
│   ├── preprocessing.zig  # Preprocessing re-exports
│   ├── preprocessing/
│   │   ├── scalers.zig    # StandardScaler / MinMaxScaler
│   │   ├── encoders.zig   # Label / Ordinal / OneHot encoders
│   │   └── binning.zig    # cut / qcut
│   ├── model_selection.zig # Model-selection re-exports
│   ├── model_selection/
│   │   └── split.zig      # train_test_split / k_fold_split
│   ├── quaternion.zig     # Quaternion math
│   ├── noise.zig          # Perlin / Simplex noise
│   ├── consts.zig         # Physical / numeric constants re-exports
│   ├── consts/
│   │   ├── num.zig        # Numeric prefixes
│   │   ├── cgs.zig        # CGS constants
│   │   ├── cgsm.zig       # CGSM constants
│   │   ├── mks.zig        # MKS constants
│   │   └── mksa.zig       # MKSA constants
│   ├── fun.zig            # Special functions re-exports
│   ├── fun/
│   │   ├── gamma.zig      # Gamma / log-gamma / factorial
│   │   ├── digamma.zig    # Digamma ψ
│   │   ├── erf.zig        # Error functions
│   │   ├── bessel.zig     # J/Y Bessel functions
│   │   ├── mod_bessel.zig # I/K modified Bessel functions
│   │   ├── misc.zig       # choose / fib / hypot
│   │   └── interp.zig     # Chebyshev / interpolation
│   ├── graph.zig          # Graph algorithms
│   ├── gm.zig             # Geometry spatial bins
│   ├── ml.zig             # Machine learning re-exports
│   ├── ml/
│   │   ├── data.zig       # Data container
│   │   ├── workspace.zig  # Per-feature statistics
│   │   ├── paramsreg.zig  # Regression parameters
│   │   ├── linreg.zig     # Linear regression
│   │   ├── kmeans.zig     # K-Means clustering
│   │   ├── knn.zig        # K-Nearest Neighbors
│   │   ├── logreg.zig     # Logistic regression
│   │   ├── svm.zig        # Support Vector Machine classifier
│   │   ├── decision_tree.zig # Classification decision tree
│   │   ├── lasso.zig      # L1-regularized regression
│   │   └── random_forest.zig # Ensemble of decision trees
│   ├── deriv.zig          # Numerical derivatives
│   ├── diff.zig           # Automatic derivative step sizing
│   ├── fft.zig            # Fast Fourier Transform
│   ├── easings.zig        # Easing functions
│   ├── inout.zig          # I/O re-exports
│   ├── inout/
│   │   └── csv.zig        # CSV reader/writer
│   ├── plot.zig           # Plotting re-exports
│   └── plot/
│       ├── plot.zig       # Plot core and HTML assembly
│       ├── layout.zig     # Layout, Axis, Annotation
│       ├── trace.zig      # Trace types, Marker, Line, JSON writer
│       ├── ml_plots.zig   # ML visualization helpers
│       └── show.zig       # save_html / show
└── examples/
    ├── vector_ops.zig     # Vector BLAS demo
    ├── matrix_ops.zig     # Dense matrix determinant / inverse / solve demo
    ├── blas_core.zig      # Symmetric/triangular BLAS demo
    ├── lapack_solve.zig   # LAPACK linear solver demo
    ├── eigen_ops.zig      # Symmetric eigenvalue decomposition demo
    ├── statistics_ops.zig # Statistics demo
    ├── metrics_ops.zig    # Metrics demo
    ├── roots_ops.zig      # Root finders demo
    ├── poly_ops.zig       # Polynomial operations demo
    ├── prime_ops.zig      # Prime utilities demo
    ├── dist_ops.zig       # Histogram demo
    ├── iter_ops.zig       # Iterator demo
    ├── preprocessing_ops.zig # Preprocessing demo
    ├── model_selection_ops.zig # Model-selection demo
    ├── quaternion_ops.zig # Quaternion demo
    ├── noise_ops.zig      # Noise demo
    ├── consts_ops.zig     # Constants demo
    ├── fun_ops.zig        # Special functions demo
    ├── graph_ops.zig      # Graph demo
    ├── gm_ops.zig         # Geometry bins demo
    ├── kmeans_ops.zig     # K-Means demo
    ├── linreg_ops.zig     # Linear regression demo
    ├── knn_ops.zig        # KNN demo
    ├── deriv_ops.zig      # Numerical derivatives demo
    ├── diff_ops.zig       # Diff demo
    ├── fft_ops.zig        # FFT demo
    ├── easings_ops.zig    # Easings demo
    ├── csv_ops.zig        # CSV demo
    ├── ml_advanced_ops.zig # Logistic regression / advanced ML demo
    └── plot_ops.zig       # Plotly HTML plotting demo
```

## 4. Build and test commands

```sh
zig build test           # run all unit tests
zig build example        # build and run examples/matrix_ops.zig
zig build example-vector # build and run examples/vector_ops.zig
zig build example-blas   # build and run examples/blas_core.zig
zig build example-lapack # build and run examples/lapack_solve.zig
zig build example-easings # run easings_ops demo
zig build example-csv    # run csv_ops demo
zig build example-ml-advanced   # run ml_advanced_ops demo
zig build example-plot   # run plot_ops example (writes zig-out/plot_ops.html)
```

## 5. Code style guidelines

- Use idiomatic Zig: `snake_case` functions, `PascalCase` types.
- Public APIs are marked with `pub`.
- Pass `std.mem.Allocator` explicitly when allocating.
- Keep `unsafe` blocks minimal; Zig safety checks handle most bounds checks.
- Write inline `test {}` blocks next to the code they test.
- Use `std.testing.allocator` for leak detection in tests.
- Use `float.approxEqAbs` / `approxEqRel` for numerical comparisons.

## 6. Reference

- `_ref/vsl/README.md` — VSL capabilities overview.
- `_ref/vsl/docs/` — VSL tutorials and architecture notes.
