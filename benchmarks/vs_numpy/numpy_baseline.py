#!/usr/bin/env python3
"""NumPy baseline timings for comparison with the zsl benchmark harness."""

import time
import numpy as np


def time_it(fn, *args, repeats=3):
    """Return the minimum elapsed wall time over `repeats` runs."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        elapsed = time.perf_counter() - start
        if elapsed < best:
            best = elapsed
    return best


def bench_dot(n=500):
    rng = np.random.default_rng(0x123456789ABCDEF0)
    a = rng.random((n, n))
    b = rng.random((n, n))
    elapsed = time_it(np.dot, a, b)
    ops = 2 * n ** 3
    print(f"np.dot ({n}x{n})          {elapsed:>10.6f} s  {ops:>12} ops  {ops / elapsed / 1e9:>8.3f} Gops/s")


def bench_solve(n=200):
    rng = np.random.default_rng(0x123456789ABCDEF0)
    a = rng.random((n, n))
    # Make diagonally dominant for a stable, well-conditioned system.
    a += n * np.eye(n)
    b = rng.random((n, 1))
    elapsed = time_it(np.linalg.solve, a, b)
    ops = 2 * n ** 3 / 3
    print(f"np.linalg.solve ({n}x{n})  {elapsed:>10.6f} s  {ops:>12} ops  {ops / elapsed / 1e9:>8.3f} Gops/s")


def bench_svd(n=100):
    rng = np.random.default_rng(0x123456789ABCDEF0)
    a = rng.random((n, n))
    elapsed = time_it(np.linalg.svd, a)
    ops = 4 * n ** 3
    print(f"np.linalg.svd ({n}x{n})    {elapsed:>10.6f} s  {ops:>12} ops  {ops / elapsed / 1e9:>8.3f} Gops/s")


def bench_eig(n=100):
    rng = np.random.default_rng(0x123456789ABCDEF0)
    # Symmetric positive definite matrix for a fair comparison with dsyev.
    m = rng.random((n, n))
    a = m.T @ m + n * np.eye(n)
    elapsed = time_it(np.linalg.eig, a)
    ops = 4 * n ** 3
    print(f"np.linalg.eig ({n}x{n})    {elapsed:>10.6f} s  {ops:>12} ops  {ops / elapsed / 1e9:>8.3f} Gops/s")


if __name__ == "__main__":
    print("=== NumPy Baseline Benchmarks ===")
    bench_dot()
    bench_solve()
    bench_svd()
    bench_eig()
