const std = @import("std");
const la = @import("../la.zig");
const util = @import("../util.zig");
const Error = @import("../errors.zig").Error;
const Matrix = la.Matrix;
const Vector = la.Vector;

/// Jacobi eigenvalue decomposition of a real symmetric matrix.
///
/// On input, `a` must be symmetric and square. On output, `a` is overwritten
/// with a diagonal matrix whose diagonal entries are the eigenvalues. `q` is
/// overwritten with the orthogonal matrix of eigenvectors (columns). `v`
/// receives the eigenvalues.
///
/// Matches the algorithm in VSL's `la/jacobi.v`.
pub fn jacobi(
    q: *Matrix(f64),
    v: *Vector(f64),
    a: *Matrix(f64),
    allocator: std.mem.Allocator,
) Error!void {
    _ = util.Float(f64);
    const n = a.rows;
    if (a.rows != a.cols) return error.ShapeMismatch;
    if (q.rows != n or q.cols != n) return error.ShapeMismatch;
    if (v.len != n) return error.ShapeMismatch;

    const tol = 1e-15;
    const max_iterations = 20;

    const b = try allocator.alloc(f64, n);
    defer allocator.free(b);
    const z = try allocator.alloc(f64, n);
    defer allocator.free(z);

    // Initialize Q to identity.
    @memset(q.data, 0);
    for (0..n) |i| {
        try q.set(i, i, 1.0);
    }

    // Initialize b and v to the diagonal of A.
    for (0..n) |i| {
        const val = try a.get(i, i);
        b[i] = val;
        v.data[i * v.stride] = val;
        z[i] = 0.0;
    }

    for (0..max_iterations) |_| {
        var sum: f64 = 0.0;
        for (0..n - 1) |i| {
            for (i + 1..n) |j| {
                sum += @abs(try a.get(i, j));
            }
        }
        if (sum < tol) break;

        for (0..n - 1) |i| {
            for (i + 1..n) |j| {
                const aij = try a.get(i, j);
                if (@abs(aij) < tol) continue;

                const h = v.data[j * v.stride] - v.data[i * v.stride];
                var t: f64 = 0.0;
                if (@abs(h) < tol and @abs(aij) < tol) {
                    t = 1.0;
                } else {
                    const theta = 0.5 * h / aij;
                    t = 1.0 / (@abs(theta) + std.math.sqrt(1.0 + theta * theta));
                    if (theta < 0.0) t = -t;
                }

                const c = 1.0 / std.math.sqrt(1.0 + t * t);
                const s = t * c;

                const aii = try a.get(i, i);
                const ajj = try a.get(j, j);
                try a.set(i, i, aii - t * aij);
                try a.set(j, j, ajj + t * aij);
                v.data[i * v.stride] = try a.get(i, i);
                v.data[j * v.stride] = try a.get(j, j);

                try a.set(i, j, 0.0);
                try a.set(j, i, 0.0);

                for (0..n) |k| {
                    if (k != i and k != j) {
                        const aik = try a.get(i, k);
                        const ajk = try a.get(j, k);
                        try a.set(i, k, c * aik - s * ajk);
                        try a.set(j, k, c * ajk + s * aik);
                        try a.set(k, i, try a.get(i, k));
                        try a.set(k, j, try a.get(j, k));
                    }
                }

                for (0..n) |k| {
                    const qik = try q.get(k, i);
                    const qjk = try q.get(k, j);
                    try q.set(k, i, c * qik - s * qjk);
                    try q.set(k, j, c * qjk + s * qik);
                }
            }
        }
    }

    for (0..n) |i| {
        try a.set(i, i, v.data[i * v.stride]);
        for (0..n) |j| {
            if (i != j) {
                try a.set(i, j, 0.0);
            }
        }
    }

    var sum: f64 = 0.0;
    for (0..n - 1) |i| {
        for (i + 1..n) |j| {
            sum += @abs(try a.get(i, j));
        }
    }
    if (sum >= tol) return error.NotConverged;
}

test "jacobi diagonalizes a symmetric matrix" {
    const T = f64;
    const M = Matrix(T);
    const V = Vector(T);
    const blas = @import("../blas.zig");
    const float = @import("../float.zig");

    var a = try M.fromRowSlice(std.testing.allocator, 3, 3, &[_]T{
        2.0, 1.0, 1.0,
        1.0, 3.0, 2.0,
        1.0, 2.0, 4.0,
    });
    defer a.deinit(std.testing.allocator);

    var q = try M.init(std.testing.allocator, 3, 3);
    defer q.deinit(std.testing.allocator);
    var v = try V.init(std.testing.allocator, 3);
    defer v.deinit(std.testing.allocator);

    try jacobi(&q, &v, &a, std.testing.allocator);

    // Build diagonal matrix from eigenvalues.
    var diag = try M.init(std.testing.allocator, 3, 3);
    defer diag.deinit(std.testing.allocator);
    for (0..3) |i| {
        try diag.set(i, i, try v.get(i));
    }

    // Reconstruct A = Q * Λ * Q^T.
    var q_lambda = try M.init(std.testing.allocator, 3, 3);
    defer q_lambda.deinit(std.testing.allocator);
    try blas.gemm(T, .no_trans, .no_trans, 1.0, q, diag, 0.0, &q_lambda);

    var reconstructed = try M.init(std.testing.allocator, 3, 3);
    defer reconstructed.deinit(std.testing.allocator);
    try blas.gemm(T, .no_trans, .trans, 1.0, q_lambda, q, 0.0, &reconstructed);

    const original = &[_]T{
        2.0, 1.0, 1.0,
        1.0, 3.0, 2.0,
        1.0, 2.0, 4.0,
    };
    for (0..3) |i| {
        for (0..3) |j| {
            try std.testing.expect(float.approxEqAbs(T, try reconstructed.get(i, j), original[i * 3 + j], 1e-12));
        }
    }
}

test "jacobi eigenvectors are orthonormal" {
    const T = f64;
    const M = Matrix(T);
    const V = Vector(T);
    const blas = @import("../blas.zig");
    const float = @import("../float.zig");

    var a = try M.fromRowSlice(std.testing.allocator, 3, 3, &[_]T{
        2.0, 1.0, 1.0,
        1.0, 3.0, 2.0,
        1.0, 2.0, 4.0,
    });
    defer a.deinit(std.testing.allocator);
    var q = try M.init(std.testing.allocator, 3, 3);
    defer q.deinit(std.testing.allocator);
    var v = try V.init(std.testing.allocator, 3);
    defer v.deinit(std.testing.allocator);

    try jacobi(&q, &v, &a, std.testing.allocator);

    var qtq = try M.init(std.testing.allocator, 3, 3);
    defer qtq.deinit(std.testing.allocator);
    try blas.gemm(T, .trans, .no_trans, 1.0, q, q, 0.0, &qtq);

    for (0..3) |i| {
        for (0..3) |j| {
            const expected = if (i == j) @as(T, 1.0) else @as(T, 0.0);
            try std.testing.expect(float.approxEqAbs(T, try qtq.get(i, j), expected, 1e-12));
        }
    }
}
