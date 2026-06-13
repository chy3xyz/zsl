const std = @import("std");
const la = @import("../la.zig");
const Error = @import("../errors.zig").Error;
const Matrix = la.Matrix;

/// Unblocked Cholesky factorization of an n×n symmetric positive definite
/// matrix A stored row-major in `a` with leading dimension `lda`.
///
/// On exit, the lower triangle (including the diagonal) of A contains L such
/// that A = L * L^T. The upper triangle is left unchanged.
pub fn dpotrf(n: usize, a: []f64, lda: usize) Error!void {
    if (n == 0) return;
    if (a.len < (n - 1) * lda + n) return error.InvalidDimension;

    for (0..n) |j| {
        var ajj = a[j * lda + j];
        for (0..j) |k| {
            const val = a[j * lda + k];
            ajj -= val * val;
        }
        if (ajj <= 0.0 or std.math.isNan(ajj)) {
            a[j * lda + j] = ajj;
            return error.NotPositiveDefinite;
        }
        ajj = std.math.sqrt(ajj);
        a[j * lda + j] = ajj;

        if (j < n - 1) {
            for (j + 1..n) |k| {
                var val = a[k * lda + j];
                for (0..j) |i| {
                    val -= a[k * lda + i] * a[j * lda + i];
                }
                a[k * lda + j] = val / ajj;
            }
        }
    }
}

/// Compute the Cholesky factor L of a symmetric positive definite matrix A.
/// Returns a newly allocated n×n matrix whose lower triangle contains L.
pub fn cholesky(allocator: std.mem.Allocator, a: Matrix(f64)) Error!Matrix(f64) {
    if (a.rows != a.cols) return error.ShapeMismatch;

    var l = try Matrix(f64).init(allocator, a.rows, a.cols);
    errdefer l.deinit(allocator);

    // Copy only the lower triangle of A; zero the upper triangle so the
    // result is explicitly lower triangular.
    for (0..a.rows) |i| {
        for (0..a.cols) |j| {
            const val = if (i >= j) try a.get(i, j) else 0.0;
            try l.set(i, j, val);
        }
    }

    try dpotrf(l.rows, l.data, l.cols);
    return l;
}

test "dpotrf factorizes a 2x2 SPD matrix" {
    const T = f64;
    const M = Matrix(T);
    const float = @import("../float.zig");
    const blas = @import("../blas.zig");

    var a = try M.fromRowSlice(std.testing.allocator, 2, 2, &[_]T{
        4.0,  12.0,
        12.0, 37.0,
    });
    defer a.deinit(std.testing.allocator);

    try dpotrf(2, a.data, a.cols);

    // Expected lower triangle: L = [[2, 0], [6, 1]].
    try std.testing.expect(float.approxEqAbs(T, try a.get(0, 0), 2.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try a.get(1, 0), 6.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try a.get(1, 1), 1.0, 1e-12));

    // dpotrf leaves the upper triangle untouched, so copy the lower triangle
    // into a clean matrix before reconstructing A = L * L^T.
    var l = try M.init(std.testing.allocator, 2, 2);
    defer l.deinit(std.testing.allocator);
    for (0..2) |i| {
        for (0..i + 1) |j| {
            try l.set(i, j, try a.get(i, j));
        }
    }

    var ll = try M.init(std.testing.allocator, 2, 2);
    defer ll.deinit(std.testing.allocator);
    try blas.gemm(T, .no_trans, .trans, 1.0, l, l, 0.0, &ll);

    const original = &[_]T{ 4.0, 12.0, 12.0, 37.0 };
    for (0..2) |i| {
        for (0..2) |j| {
            try std.testing.expect(float.approxEqAbs(T, try ll.get(i, j), original[i * 2 + j], 1e-12));
        }
    }
}

test "dpotrf factorizes a 3x3 SPD matrix" {
    const T = f64;
    const M = Matrix(T);
    const float = @import("../float.zig");
    const blas = @import("../blas.zig");

    var a = try M.fromRowSlice(std.testing.allocator, 3, 3, &[_]T{
        4.0, 2.0, 1.0,
        2.0, 5.0, 3.0,
        1.0, 3.0, 6.0,
    });
    defer a.deinit(std.testing.allocator);

    try dpotrf(3, a.data, a.cols);

    // Copy lower triangle into a clean matrix before reconstructing.
    var l = try M.init(std.testing.allocator, 3, 3);
    defer l.deinit(std.testing.allocator);
    for (0..3) |i| {
        for (0..i + 1) |j| {
            try l.set(i, j, try a.get(i, j));
        }
    }

    var ll = try M.init(std.testing.allocator, 3, 3);
    defer ll.deinit(std.testing.allocator);
    try blas.gemm(T, .no_trans, .trans, 1.0, l, l, 0.0, &ll);

    const original = &[_]T{
        4.0, 2.0, 1.0,
        2.0, 5.0, 3.0,
        1.0, 3.0, 6.0,
    };
    for (0..3) |i| {
        for (0..3) |j| {
            try std.testing.expect(float.approxEqAbs(T, try ll.get(i, j), original[i * 3 + j], 1e-10));
        }
    }
}

test "cholesky wrapper returns lower triangular L" {
    const T = f64;
    const M = Matrix(T);
    const float = @import("../float.zig");
    const blas = @import("../blas.zig");

    var a = try M.fromRowSlice(std.testing.allocator, 3, 3, &[_]T{
        2.0, 1.0, 1.0,
        1.0, 3.0, 2.0,
        1.0, 2.0, 4.0,
    });
    defer a.deinit(std.testing.allocator);

    var l = try cholesky(std.testing.allocator, a);
    defer l.deinit(std.testing.allocator);

    // Upper triangle is explicitly zeroed.
    for (0..3) |i| {
        for (i + 1..3) |j| {
            try std.testing.expectEqual(@as(T, 0.0), try l.get(i, j));
        }
    }

    var ll = try M.init(std.testing.allocator, 3, 3);
    defer ll.deinit(std.testing.allocator);
    try blas.gemm(T, .no_trans, .trans, 1.0, l, l, 0.0, &ll);

    for (0..3) |i| {
        for (0..3) |j| {
            try std.testing.expect(float.approxEqAbs(T, try ll.get(i, j), try a.get(i, j), 1e-10));
        }
    }
}

test "dpotrf detects non-positive-definite matrix" {
    const T = f64;
    const M = Matrix(T);

    var a = try M.fromRowSlice(std.testing.allocator, 2, 2, &[_]T{
        1.0, 2.0,
        2.0, 1.0,
    });
    defer a.deinit(std.testing.allocator);

    try std.testing.expectError(error.NotPositiveDefinite, dpotrf(2, a.data, a.cols));
}
