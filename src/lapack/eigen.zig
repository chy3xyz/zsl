const std = @import("std");
const la = @import("../la.zig");
const Error = @import("../errors.zig").Error;
const Matrix = la.Matrix;
const Vector = la.Vector;

/// Symmetric eigenvalue decomposition via Jacobi rotations.
///
/// The input matrix `a` is expected to be symmetric; only its symmetric part
/// is considered if the input is not exactly symmetric.
///
/// The caller owns both `eigenvalues` and `eigenvectors`. Eigenvectors are
/// stored as columns of the returned matrix: A * v_i ≈ λ_i * v_i.
pub fn dsyev(allocator: std.mem.Allocator, a: Matrix(f64)) Error!struct {
    eigenvalues: []f64,
    eigenvectors: Matrix(f64),
} {
    if (a.rows != a.cols) return error.ShapeMismatch;
    const n = a.rows;

    // jacobi overwrites its input matrix, so make a private copy.
    var a_copy = try Matrix(f64).fromRowSlice(allocator, n, n, a.data);
    defer a_copy.deinit(allocator);

    var q = try Matrix(f64).init(allocator, n, n);
    errdefer q.deinit(allocator);

    var v = try Vector(f64).init(allocator, n);
    errdefer v.deinit(allocator);

    try la.jacobi.jacobi(&q, &v, &a_copy, allocator);

    // Caller owns the eigenvalue slice.
    const eigenvalues = try allocator.alloc(f64, n);
    errdefer allocator.free(eigenvalues);
    for (0..n) |i| {
        eigenvalues[i] = v.data[i * v.stride];
    }

    v.deinit(allocator);
    return .{
        .eigenvalues = eigenvalues,
        .eigenvectors = q,
    };
}

test "dsyev diagonalizes a symmetric matrix" {
    const T = f64;
    const M = Matrix(T);
    const V = Vector(T);
    const float = @import("../float.zig");
    const blas = @import("../blas.zig");

    var a = try M.fromRowSlice(std.testing.allocator, 3, 3, &[_]T{
        2.0, 1.0, 1.0,
        1.0, 3.0, 2.0,
        1.0, 2.0, 4.0,
    });
    defer a.deinit(std.testing.allocator);

    var result = try dsyev(std.testing.allocator, a);
    defer std.testing.allocator.free(result.eigenvalues);
    defer result.eigenvectors.deinit(std.testing.allocator);

    // Verify A * v_i ≈ λ_i * v_i for each eigenvector.
    var av = try V.init(std.testing.allocator, 3);
    defer av.deinit(std.testing.allocator);
    var lv = try V.init(std.testing.allocator, 3);
    defer lv.deinit(std.testing.allocator);

    for (0..3) |j| {
        const lambda = result.eigenvalues[j];
        const vj = try result.eigenvectors.col(j);
        try blas.gemv(T, .no_trans, 1.0, a, vj, 0.0, &av);
        try blas.copy(T, vj, &lv);
        try blas.scal(T, lambda, &lv);
        for (0..3) |i| {
            try std.testing.expect(float.approxEqAbs(T, try av.get(i), try lv.get(i), 1e-10));
        }
    }
}

test "dsyev eigenvectors are orthonormal" {
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

    var result = try dsyev(std.testing.allocator, a);
    defer std.testing.allocator.free(result.eigenvalues);
    defer result.eigenvectors.deinit(std.testing.allocator);

    var qtq = try M.init(std.testing.allocator, 3, 3);
    defer qtq.deinit(std.testing.allocator);
    try blas.gemm(T, .trans, .no_trans, 1.0, result.eigenvectors, result.eigenvectors, 0.0, &qtq);

    for (0..3) |i| {
        for (0..3) |j| {
            const expected = if (i == j) @as(T, 1.0) else @as(T, 0.0);
            try std.testing.expect(float.approxEqAbs(T, try qtq.get(i, j), expected, 1e-12));
        }
    }
}
