const std = @import("std");
const blas = @import("../blas.zig");
const la = @import("../la.zig");
const Error = @import("../errors.zig").Error;
const helpers = @import("helpers.zig");
const Matrix = la.Matrix;

/// Unblocked QR factorization of an m×n row-major matrix A.
///
/// On exit, the upper triangle of A (including the diagonal) contains R,
/// the strict lower triangle contains the Householder vectors, and tau[i]
/// contains the scalar for the i-th reflector.
pub fn dgeqr2(m: usize, n: usize, a: []f64, lda: usize, tau: []f64) void {
    if (m == 0 or n == 0) return;
    std.debug.assert(a.len >= (m - 1) * lda + n);

    const k = @min(m, n);
    std.debug.assert(tau.len >= k);

    for (0..k) |i| {
        const tau_i = helpers.dlarfg(m - i, a[i * lda + i ..], lda);
        tau[i] = tau_i;

        if (i < n - 1) {
            const beta = a[i * lda + i];
            a[i * lda + i] = 1.0;
            helpers.dlarf(.left, m - i, n - i - 1, a[i * lda + i ..], lda, tau_i, a[i * lda + i + 1 ..], lda);
            a[i * lda + i] = beta;
        }
    }
}

/// Generate the m×n orthogonal matrix Q from the output of dgeqr2.
/// k is the number of reflectors (min(m, n)). On exit, the first m×n part of
/// A contains Q.
pub fn dorg2r(m: usize, n: usize, k: usize, a: []f64, lda: usize, tau: []const f64) void {
    if (n == 0 or k == 0) return;
    std.debug.assert(k <= n);
    std.debug.assert(a.len >= (m - 1) * lda + n);
    std.debug.assert(tau.len >= k);

    // Initialize columns k..n-1 to columns of the identity matrix.
    for (0..m) |l| {
        for (k..n) |j| {
            a[l * lda + j] = 0.0;
        }
    }
    for (k..n) |j| {
        a[j * lda + j] = 1.0;
    }

    var i: usize = k;
    while (i > 0) {
        i -= 1;

        if (i < n - 1) {
            a[i * lda + i] = 1.0;
            helpers.dlarf(.left, m - i, n - i - 1, a[i * lda + i ..], lda, tau[i], a[i * lda + i + 1 ..], lda);
        }

        if (i < m - 1) {
            const neg_tau = -tau[i];
            for (i + 1..m) |l| {
                a[l * lda + i] *= neg_tau;
            }
        }

        a[i * lda + i] = 1.0 - tau[i];
        for (0..i) |l| {
            a[l * lda + i] = 0.0;
        }
    }
}

/// Convenience wrapper: compute the thin QR factorization of A.
/// Returns Q (m×min(m,n)) and R (min(m,n)×n). The caller must deinit both.
pub fn qr(allocator: std.mem.Allocator, a: Matrix(f64)) Error!struct { q: Matrix(f64), r: Matrix(f64) } {
    const m = a.rows;
    const n = a.cols;
    const k = @min(m, n);

    var work = try Matrix(f64).init(allocator, m, n);
    errdefer work.deinit(allocator);
    @memcpy(work.data, a.data);

    const tau = try allocator.alloc(f64, k);
    errdefer allocator.free(tau);

    dgeqr2(m, n, work.data, work.cols, tau);

    var r = try Matrix(f64).init(allocator, k, n);
    errdefer r.deinit(allocator);
    for (0..k) |i| {
        for (i..n) |j| {
            r.data[i * r.row_stride + j * r.col_stride] = work.data[i * work.row_stride + j * work.col_stride];
        }
    }

    dorg2r(m, k, k, work.data, work.cols, tau);

    var q = try Matrix(f64).init(allocator, m, k);
    errdefer q.deinit(allocator);
    for (0..m) |i| {
        for (0..k) |j| {
            q.data[i * q.row_stride + j * q.col_stride] = work.data[i * work.row_stride + j * work.col_stride];
        }
    }

    allocator.free(tau);
    work.deinit(allocator);

    return .{ .q = q, .r = r };
}

test "dgeqr2 and dorg2r reconstruct matrix" {
    const T = f64;
    const M = Matrix(T);
    const float = @import("../float.zig");

    const m = 4;
    const n = 3;
    const k = @min(m, n);

    var a = try M.fromRowSlice(std.testing.allocator, m, n, &[_]T{
        1.0, 2.0,  3.0,
        4.0, 5.0,  6.0,
        7.0, 8.0,  10.0,
        9.0, 11.0, 12.0,
    });
    defer a.deinit(std.testing.allocator);

    var original = try M.fromRowSlice(std.testing.allocator, m, n, a.data);
    defer original.deinit(std.testing.allocator);

    const tau = try std.testing.allocator.alloc(T, k);
    defer std.testing.allocator.free(tau);

    dgeqr2(m, n, a.data, a.cols, tau);

    // Extract R from the upper k×n block.
    var r = try M.init(std.testing.allocator, k, n);
    defer r.deinit(std.testing.allocator);
    for (0..k) |i| {
        for (i..n) |j| {
            try r.set(i, j, try a.get(i, j));
        }
    }

    dorg2r(m, n, k, a.data, a.cols, tau);

    // Verify Q^T * Q = I.
    var qtq = try M.init(std.testing.allocator, k, k);
    defer qtq.deinit(std.testing.allocator);
    try blas.gemm(T, .trans, .no_trans, 1.0, a, a, 0.0, &qtq);
    for (0..k) |i| {
        for (0..k) |j| {
            const expected = if (i == j) @as(T, 1.0) else @as(T, 0.0);
            try std.testing.expect(float.approxEqAbs(T, try qtq.get(i, j), expected, 1e-10));
        }
    }

    // Verify Q * R = A.
    var qr_prod = try M.init(std.testing.allocator, m, n);
    defer qr_prod.deinit(std.testing.allocator);
    try blas.gemm(T, .no_trans, .no_trans, 1.0, a, r, 0.0, &qr_prod);
    for (0..m) |i| {
        for (0..n) |j| {
            try std.testing.expect(float.approxEqAbs(T, try qr_prod.get(i, j), try original.get(i, j), 1e-9));
        }
    }
}

test "qr wrapper on wide matrix" {
    const T = f64;
    const M = Matrix(T);
    const float = @import("../float.zig");

    var a = try M.fromRowSlice(std.testing.allocator, 2, 3, &[_]T{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer a.deinit(std.testing.allocator);

    var result = try qr(std.testing.allocator, a);
    defer result.q.deinit(std.testing.allocator);
    defer result.r.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), result.q.rows);
    try std.testing.expectEqual(@as(usize, 2), result.q.cols);
    try std.testing.expectEqual(@as(usize, 2), result.r.rows);
    try std.testing.expectEqual(@as(usize, 3), result.r.cols);

    // Q^T * Q = I.
    var qtq = try M.init(std.testing.allocator, result.q.cols, result.q.cols);
    defer qtq.deinit(std.testing.allocator);
    try blas.gemm(T, .trans, .no_trans, 1.0, result.q, result.q, 0.0, &qtq);
    for (0..qtq.rows) |i| {
        for (0..qtq.cols) |j| {
            const expected = if (i == j) @as(T, 1.0) else @as(T, 0.0);
            try std.testing.expect(float.approxEqAbs(T, try qtq.get(i, j), expected, 1e-10));
        }
    }

    // Q * R = A.
    var qr_prod = try M.init(std.testing.allocator, a.rows, a.cols);
    defer qr_prod.deinit(std.testing.allocator);
    try blas.gemm(T, .no_trans, .no_trans, 1.0, result.q, result.r, 0.0, &qr_prod);
    for (0..a.rows) |i| {
        for (0..a.cols) |j| {
            try std.testing.expect(float.approxEqAbs(T, try qr_prod.get(i, j), try a.get(i, j), 1e-9));
        }
    }
}

test "qr wrapper on square matrix" {
    const T = f64;
    const M = Matrix(T);
    const float = @import("../float.zig");

    var a = try M.fromRowSlice(std.testing.allocator, 3, 3, &[_]T{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0,
    });
    defer a.deinit(std.testing.allocator);

    var result = try qr(std.testing.allocator, a);
    defer result.q.deinit(std.testing.allocator);
    defer result.r.deinit(std.testing.allocator);

    // Q^T * Q = I.
    var qtq = try M.init(std.testing.allocator, result.q.cols, result.q.cols);
    defer qtq.deinit(std.testing.allocator);
    try blas.gemm(T, .trans, .no_trans, 1.0, result.q, result.q, 0.0, &qtq);
    for (0..qtq.rows) |i| {
        for (0..qtq.cols) |j| {
            const expected = if (i == j) @as(T, 1.0) else @as(T, 0.0);
            try std.testing.expect(float.approxEqAbs(T, try qtq.get(i, j), expected, 1e-10));
        }
    }

    // Q * R = A.
    var qr_prod = try M.init(std.testing.allocator, a.rows, a.cols);
    defer qr_prod.deinit(std.testing.allocator);
    try blas.gemm(T, .no_trans, .no_trans, 1.0, result.q, result.r, 0.0, &qr_prod);
    for (0..a.rows) |i| {
        for (0..a.cols) |j| {
            try std.testing.expect(float.approxEqAbs(T, try qr_prod.get(i, j), try a.get(i, j), 1e-9));
        }
    }
}
