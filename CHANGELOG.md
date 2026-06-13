# Changelog

All notable changes to `zsl` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-12

### Added

- Initial public release of `zsl`, a pure-Zig scientific computing library.
- Linear algebra: dense `Vector(T)` / `Matrix(T)`, BLAS Level-1/2/3,
  LU factorization, linear solvers, symmetric eigenvalue decomposition.
- Statistics and evaluation metrics.
- Numerical utilities: root finders, polynomials, primes, histograms,
  combinatorial iterators.
- Preprocessing and model-selection primitives.
- Quaternions, Perlin/Simplex noise, physical constants.
- Special functions: gamma, digamma, error functions, Bessel functions,
  interpolation.
- Graph algorithms and geometry spatial bins.
- Fast Fourier Transform (radix-2 Cooley-Tukey).
- Machine learning: `Data(T)` container, K-Means, linear regression, KNN,
  logistic regression, SVM, decision trees, LASSO, random forests.
- Easing/tweening functions and Plotly HTML plotting.
- CSV reader/writer.
- Runnable examples for every major module.
