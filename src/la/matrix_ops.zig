const std = @import("std");
const la = @import("../la.zig");
const lapack = @import("../lapack.zig");
const util = @import("../util.zig");
const Error = @import("../errors.zig").Error;
const Matrix = la.Matrix;
const Vector = la.Vector;

fn cloneRowMajor(comptime T: type, src: Matrix(T), allocator: std.mem.Allocator) Error!Matrix(T) {
    var dst = try Matrix(T).init(allocator, src.rows, src.cols);
    for (0..src.rows) |i| {
        for (0..src.cols) |j| {
            dst.data[i * src.cols + j] = try src.get(i, j);
        }
    }
    return dst;
}

/// Determinant of a square matrix using LU factorization.
/// The input is preserved; `allocator` is used for the temporary copy.
pub fn det(comptime T: type, a: Matrix(T), allocator: std.mem.Allocator) Error!T {
    _ = util.Float(T);
    if (a.rows != a.cols) return error.ShapeMismatch;
    if (a.rows == 0) return error.InvalidDimension;

    var acpy = try cloneRowMajor(T, a, allocator);
    defer acpy.deinit(allocator);

    const ipiv = try allocator.alloc(usize, a.rows);
    defer allocator.free(ipiv);

    const ok = try lapack.lu.dgetrf(T, a.rows, a.cols, &acpy, ipiv);
    if (!ok) return error.SingularMatrix;

    var result: T = 1.0;
    var swaps: usize = 0;
    for (0..a.rows) |i| {
        if (ipiv[i] != i) swaps += 1;
        result *= acpy.data[i * acpy.cols + i];
    }
    if (swaps % 2 == 1) result = -result;
    return result;
}

/// Inverse of a small (1x1, 2x2, or 3x3) matrix using closed-form formulas.
/// Writes the result into `out` and returns the determinant.
pub fn inverse_small(comptime T: type, a: Matrix(T), out: *Matrix(T), tol: T) Error!T {
    _ = util.Float(T);
    if (a.rows != a.cols or out.rows != a.rows or out.cols != a.cols) {
        return error.ShapeMismatch;
    }
    const n = a.rows;

    if (n == 1) {
        const d = try a.get(0, 0);
        if (@abs(d) < tol) return error.SingularMatrix;
        try out.set(0, 0, 1.0 / d);
        return d;
    }

    if (n == 2) {
        const a00 = try a.get(0, 0);
        const a01 = try a.get(0, 1);
        const a10 = try a.get(1, 0);
        const a11 = try a.get(1, 1);
        const d = a00 * a11 - a01 * a10;
        if (@abs(d) < tol) return error.SingularMatrix;
        try out.set(0, 0, a11 / d);
        try out.set(0, 1, -a01 / d);
        try out.set(1, 0, -a10 / d);
        try out.set(1, 1, a00 / d);
        return d;
    }

    if (n == 3) {
        const a00 = try a.get(0, 0);
        const a01 = try a.get(0, 1);
        const a02 = try a.get(0, 2);
        const a10 = try a.get(1, 0);
        const a11 = try a.get(1, 1);
        const a12 = try a.get(1, 2);
        const a20 = try a.get(2, 0);
        const a21 = try a.get(2, 1);
        const a22 = try a.get(2, 2);

        const d = a00 * (a11 * a22 - a12 * a21) -
            a01 * (a10 * a22 - a12 * a20) +
            a02 * (a10 * a21 - a11 * a20);
        if (@abs(d) < tol) return error.SingularMatrix;

        try out.set(0, 0, (a11 * a22 - a12 * a21) / d);
        try out.set(0, 1, (a02 * a21 - a01 * a22) / d);
        try out.set(0, 2, (a01 * a12 - a02 * a11) / d);
        try out.set(1, 0, (a12 * a20 - a10 * a22) / d);
        try out.set(1, 1, (a00 * a22 - a02 * a20) / d);
        try out.set(1, 2, (a02 * a10 - a00 * a12) / d);
        try out.set(2, 0, (a10 * a21 - a11 * a20) / d);
        try out.set(2, 1, (a01 * a20 - a00 * a21) / d);
        try out.set(2, 2, (a00 * a11 - a01 * a10) / d);
        return d;
    }

    return error.InvalidDimension;
}

/// Inverse of a square matrix using LU factorization.
/// The returned matrix is allocated with `allocator` and must be freed by the caller.
pub fn inverse(comptime T: type, a: Matrix(T), allocator: std.mem.Allocator) Error!Matrix(T) {
    _ = util.Float(T);
    if (a.rows != a.cols) return error.ShapeMismatch;
    const n = a.rows;

    var acpy = try cloneRowMajor(T, a, allocator);
    defer acpy.deinit(allocator);

    const ipiv = try allocator.alloc(usize, n);
    defer allocator.free(ipiv);

    const ok = try lapack.lu.dgetrf(T, n, n, &acpy, ipiv);
    if (!ok) return error.SingularMatrix;

    var inv = try Matrix(T).init(allocator, n, n);
    errdefer inv.deinit(allocator);
    @memset(inv.data, 0);
    for (0..n) |i| {
        try inv.set(i, i, 1.0);
    }

    try lapack.lu.dgetrs(T, .no_trans, acpy, ipiv, &inv);
    return inv;
}

/// Solve A*x = b for a square matrix A and vector b.
/// The result is written into `x`, which must already be allocated.
pub fn solve(
    comptime T: type,
    a: Matrix(T),
    b: Vector(T),
    x: *Vector(T),
    allocator: std.mem.Allocator,
) Error!void {
    _ = util.Float(T);
    if (a.rows != a.cols) return error.ShapeMismatch;
    if (a.rows != b.len or a.rows != x.len) return error.ShapeMismatch;
    const n = a.rows;

    for (0..b.len) |i| {
        x.data[i * x.stride] = b.data[i * b.stride];
    }

    var acpy = try cloneRowMajor(T, a, allocator);
    defer acpy.deinit(allocator);

    const ipiv = try allocator.alloc(usize, n);
    defer allocator.free(ipiv);

    const ok = try lapack.lu.dgetrf(T, n, n, &acpy, ipiv);
    if (!ok) return error.SingularMatrix;

    var x_mat = Matrix(T){
        .data = x.data,
        .rows = n,
        .cols = 1,
        .row_stride = x.stride,
        .col_stride = 1,
    };
    try lapack.lu.dgetrs(T, .no_trans, acpy, ipiv, &x_mat);
}

test "det of 2x2 and 3x3" {
    const T = f64;
    const M = Matrix(T);
    const float = @import("../float.zig");
    var a2 = try M.fromRowSlice(std.testing.allocator, 2, 2, &[_]T{
        1.0, 2.0,
        3.0, 4.0,
    });
    defer a2.deinit(std.testing.allocator);
    try std.testing.expect(float.approxEqAbs(T, try det(T, a2, std.testing.allocator), -2.0, 1e-12));

    var a3 = try M.fromRowSlice(std.testing.allocator, 3, 3, &[_]T{
        1.0, 2.0, 3.0,
        0.0, 1.0, 4.0,
        5.0, 6.0, 0.0,
    });
    defer a3.deinit(std.testing.allocator);
    try std.testing.expect(float.approxEqAbs(T, try det(T, a3, std.testing.allocator), 1.0, 1e-12));
}

test "inverse_small round trip" {
    const T = f64;
    const M = Matrix(T);
    const float = @import("../float.zig");
    var a = try M.fromRowSlice(std.testing.allocator, 2, 2, &[_]T{
        4.0, 7.0,
        2.0, 6.0,
    });
    defer a.deinit(std.testing.allocator);
    var out = try M.init(std.testing.allocator, 2, 2);
    defer out.deinit(std.testing.allocator);

    const d = try inverse_small(T, a, &out, 1e-12);
    try std.testing.expect(float.approxEqAbs(T, d, 10.0, 1e-12));

    var prod = try M.init(std.testing.allocator, 2, 2);
    defer prod.deinit(std.testing.allocator);
    try @import("../blas.zig").gemm(T, .no_trans, .no_trans, 1.0, a, out, 0.0, &prod);
    try std.testing.expect(float.approxEqAbs(T, try prod.get(0, 0), 1.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try prod.get(1, 1), 1.0, 1e-12));
}

test "inverse round trip" {
    const T = f64;
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 3, 3, &[_]T{
        2.0, 1.0, 1.0,
        1.0, 3.0, 2.0,
        1.0, 0.0, 0.0,
    });
    defer a.deinit(std.testing.allocator);

    var inv = try inverse(T, a, std.testing.allocator);
    defer inv.deinit(std.testing.allocator);

    var prod = try M.init(std.testing.allocator, 3, 3);
    defer prod.deinit(std.testing.allocator);
    try @import("../blas.zig").gemm(T, .no_trans, .no_trans, 1.0, a, inv, 0.0, &prod);

    const float = @import("../float.zig");
    for (0..3) |i| {
        for (0..3) |j| {
            const expected = if (i == j) @as(T, 1.0) else @as(T, 0.0);
            try std.testing.expect(float.approxEqAbs(T, try prod.get(i, j), expected, 1e-10));
        }
    }
}

test "solve linear system" {
    const T = f64;
    const M = Matrix(T);
    const V = Vector(T);
    var a = try M.fromRowSlice(std.testing.allocator, 3, 3, &[_]T{
        2.0, 1.0, 1.0,
        1.0, 3.0, 2.0,
        1.0, 0.0, 0.0,
    });
    defer a.deinit(std.testing.allocator);
    var b = try V.fromSlice(std.testing.allocator, &[_]T{ 4.0, 6.0, 1.0 });
    defer b.deinit(std.testing.allocator);
    var x = try V.init(std.testing.allocator, 3);
    defer x.deinit(std.testing.allocator);

    try solve(T, a, b, &x, std.testing.allocator);
    try std.testing.expectEqual(@as(T, 1.0), try x.get(0));
    try std.testing.expectEqual(@as(T, 1.0), try x.get(1));
    try std.testing.expectEqual(@as(T, 1.0), try x.get(2));
}
