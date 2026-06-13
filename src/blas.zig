const std = @import("std");
const util = @import("util.zig");
const Error = @import("errors.zig").Error;
const Vector = @import("la.zig").Vector;
const Matrix = @import("la.zig").Matrix;

pub const types = @import("blas/types.zig");
pub const complex = @import("blas/complex.zig");
pub const Transpose = types.Transpose;
pub const Uplo = types.Uplo;
pub const Side = types.Side;
pub const Diagonal = types.Diagonal;

test {
    _ = types;
    _ = complex;
}

fn checkSameLengthVectors(comptime T: type, a: Vector(T), b: Vector(T)) Error!void {
    _ = util.Float(T);
    try util.checkSameLength(a.len, b.len);
}

pub fn axpy(comptime T: type, alpha: T, x: Vector(T), y: *Vector(T)) Error!void {
    _ = util.Float(T);
    try checkSameLengthVectors(T, x, y.*);
    for (0..y.len) |i| {
        y.data[i * y.stride] += alpha * x.data[i * x.stride];
    }
}

pub fn dot(comptime T: type, x: Vector(T), y: Vector(T)) Error!T {
    _ = util.Float(T);
    try checkSameLengthVectors(T, x, y);
    var sum: T = 0;
    for (0..x.len) |i| {
        sum += x.data[i * x.stride] * y.data[i * y.stride];
    }
    return sum;
}

pub fn nrm2(comptime T: type, x: Vector(T)) Error!T {
    _ = util.Float(T);
    var sum: T = 0;
    for (0..x.len) |i| {
        const v = x.data[i * x.stride];
        sum += v * v;
    }
    return @sqrt(sum);
}

pub fn scal(comptime T: type, alpha: T, x: *Vector(T)) Error!void {
    _ = util.Float(T);
    for (0..x.len) |i| {
        x.data[i * x.stride] *= alpha;
    }
}

pub fn copy(comptime T: type, x: Vector(T), y: *Vector(T)) Error!void {
    _ = util.Float(T);
    try checkSameLengthVectors(T, x, y.*);
    for (0..x.len) |i| {
        y.data[i * y.stride] = x.data[i * x.stride];
    }
}

pub fn swap(comptime T: type, x: *Vector(T), y: *Vector(T)) Error!void {
    _ = util.Float(T);
    try checkSameLengthVectors(T, x.*, y.*);
    for (0..x.len) |i| {
        const tmp = x.data[i * x.stride];
        x.data[i * x.stride] = y.data[i * y.stride];
        y.data[i * y.stride] = tmp;
    }
}

pub fn asum(comptime T: type, x: Vector(T)) Error!T {
    _ = util.Float(T);
    var sum: T = 0;
    for (0..x.len) |i| {
        sum += @abs(x.data[i * x.stride]);
    }
    return sum;
}

pub fn iamax(comptime T: type, x: Vector(T)) Error!usize {
    _ = util.Float(T);
    if (x.len == 0) return error.InvalidDimension;
    var max_idx: usize = 0;
    var max_val: T = @abs(x.data[0]);
    for (1..x.len) |i| {
        const v = @abs(x.data[i * x.stride]);
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }
    return max_idx;
}

test "axpy adds scaled vector" {
    const T = f64;
    const V = Vector(T);
    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 2.0, 3.0 });
    defer x.deinit(std.testing.allocator);
    var y = try V.fromSlice(std.testing.allocator, &[_]T{ 4.0, 5.0, 6.0 });
    defer y.deinit(std.testing.allocator);
    try axpy(T, 2.0, x, &y);
    const float = @import("float.zig");
    try std.testing.expect(float.approxEqAbs(T, try y.get(0), 6.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try y.get(1), 9.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try y.get(2), 12.0, 1e-12));
}

test "dot product" {
    const T = f64;
    const V = Vector(T);
    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 2.0, 3.0 });
    defer x.deinit(std.testing.allocator);
    var y = try V.fromSlice(std.testing.allocator, &[_]T{ 4.0, 5.0, 6.0 });
    defer y.deinit(std.testing.allocator);
    const result = try dot(T, x, y);
    const float = @import("float.zig");
    try std.testing.expect(float.approxEqAbs(T, result, 32.0, 1e-12));
}

test "nrm2 and asum" {
    const T = f64;
    const V = Vector(T);
    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 3.0, 4.0 });
    defer x.deinit(std.testing.allocator);
    const float = @import("float.zig");
    try std.testing.expect(float.approxEqAbs(T, try nrm2(T, x), 5.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try asum(T, x), 7.0, 1e-12));
}

test "scal copy swap iamax" {
    const T = f32;
    const V = Vector(T);
    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 2.0, 3.0 });
    defer x.deinit(std.testing.allocator);
    var y = try V.fromSlice(std.testing.allocator, &[_]T{ 0.0, 0.0, 0.0 });
    defer y.deinit(std.testing.allocator);

    try scal(T, 2.0, &x);
    try std.testing.expectEqual(@as(T, 2.0), try x.get(0));

    try copy(T, x, &y);
    try std.testing.expectEqual(@as(T, 2.0), try y.get(0));

    var a = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 2.0, 3.0 });
    defer a.deinit(std.testing.allocator);
    var b = try V.fromSlice(std.testing.allocator, &[_]T{ 4.0, 5.0, 6.0 });
    defer b.deinit(std.testing.allocator);
    try swap(T, &a, &b);
    try std.testing.expectEqual(@as(T, 4.0), try a.get(0));
    try std.testing.expectEqual(@as(T, 1.0), try b.get(0));

    try std.testing.expectEqual(@as(usize, 2), try iamax(T, b));
}

test "BLAS shape mismatch" {
    const T = f64;
    const V = Vector(T);
    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 2.0 });
    defer x.deinit(std.testing.allocator);
    var y = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 2.0, 3.0 });
    defer y.deinit(std.testing.allocator);
    try std.testing.expectError(error.ShapeMismatch, dot(T, x, y));
}

pub fn gemv(
    comptime T: type,
    trans_a: Transpose,
    alpha: T,
    a: Matrix(T),
    x: Vector(T),
    beta: T,
    y: *Vector(T),
) Error!void {
    _ = util.Float(T);

    const m = a.rows;
    const n = a.cols;

    switch (trans_a) {
        .no_trans => {
            if (m != y.len or n != x.len) return error.ShapeMismatch;
        },
        .trans, .conj_trans => {
            if (n != y.len or m != x.len) return error.ShapeMismatch;
        },
    }

    // Form y = beta * y
    if (beta == 0) {
        for (0..y.len) |i| {
            y.data[i * y.stride] = 0;
        }
    } else if (beta != 1) {
        for (0..y.len) |i| {
            y.data[i * y.stride] *= beta;
        }
    }

    if (alpha == 0) return;

    switch (trans_a) {
        .no_trans => {
            for (0..m) |i| {
                var sum: T = 0;
                for (0..n) |j| {
                    sum += a.data[i * a.row_stride + j * a.col_stride] * x.data[j * x.stride];
                }
                y.data[i * y.stride] += alpha * sum;
            }
        },
        .trans, .conj_trans => {
            for (0..m) |i| {
                const tmp = alpha * x.data[i * x.stride];
                for (0..n) |j| {
                    y.data[j * y.stride] += tmp * a.data[i * a.row_stride + j * a.col_stride];
                }
            }
        },
    }
}

test "gemv no_trans f64" {
    const T = f64;
    const V = Vector(T);
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 2, 3, &[_]T{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer a.deinit(std.testing.allocator);
    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 0.5, 2.0 });
    defer x.deinit(std.testing.allocator);
    var y = try V.fromSlice(std.testing.allocator, &[_]T{ 0.0, 0.0 });
    defer y.deinit(std.testing.allocator);

    try gemv(T, .no_trans, 1.0, a, x, 0.0, &y);
    const float = @import("float.zig");
    try std.testing.expect(float.approxEqAbs(T, try y.get(0), 8.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try y.get(1), 18.5, 1e-12));
}

test "gemv trans f64" {
    const T = f64;
    const V = Vector(T);
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 2, 3, &[_]T{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    });
    defer a.deinit(std.testing.allocator);
    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 0.5 });
    defer x.deinit(std.testing.allocator);
    var y = try V.fromSlice(std.testing.allocator, &[_]T{ 0.0, 0.0, 0.0 });
    defer y.deinit(std.testing.allocator);

    try gemv(T, .trans, 1.0, a, x, 0.0, &y);
    const float = @import("float.zig");
    try std.testing.expect(float.approxEqAbs(T, try y.get(0), 3.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try y.get(1), 4.5, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try y.get(2), 6.0, 1e-12));
}

test "gemv beta accumulation" {
    const T = f64;
    const V = Vector(T);
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 2, 2, &[_]T{
        1.0, 2.0,
        3.0, 4.0,
    });
    defer a.deinit(std.testing.allocator);
    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 1.0 });
    defer x.deinit(std.testing.allocator);
    var y = try V.fromSlice(std.testing.allocator, &[_]T{ 10.0, 20.0 });
    defer y.deinit(std.testing.allocator);

    try gemv(T, .no_trans, 1.0, a, x, 2.0, &y);
    const float = @import("float.zig");
    try std.testing.expect(float.approxEqAbs(T, try y.get(0), 23.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try y.get(1), 47.0, 1e-12));
}

test "gemv shape mismatch" {
    const T = f64;
    const V = Vector(T);
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 2, 3, &[_]T{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer a.deinit(std.testing.allocator);
    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 0.5 });
    defer x.deinit(std.testing.allocator);
    var y = try V.fromSlice(std.testing.allocator, &[_]T{ 0.0, 0.0 });
    defer y.deinit(std.testing.allocator);

    try std.testing.expectError(error.ShapeMismatch, gemv(T, .no_trans, 1.0, a, x, 0.0, &y));
}

pub fn ger(
    comptime T: type,
    alpha: T,
    x: Vector(T),
    y: Vector(T),
    a: *Matrix(T),
) Error!void {
    _ = util.Float(T);
    if (a.rows != x.len or a.cols != y.len) return error.ShapeMismatch;
    if (alpha == 0) return;

    for (0..a.rows) |i| {
        const tmp = alpha * x.data[i * x.stride];
        for (0..a.cols) |j| {
            a.data[i * a.row_stride + j * a.col_stride] += tmp * y.data[j * y.stride];
        }
    }
}

test "ger rank-one update" {
    const T = f64;
    const V = Vector(T);
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 2, 3, &[_]T{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer a.deinit(std.testing.allocator);
    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 2.0, 3.0 });
    defer x.deinit(std.testing.allocator);
    var y = try V.fromSlice(std.testing.allocator, &[_]T{ 0.5, 1.0, 1.5 });
    defer y.deinit(std.testing.allocator);

    try ger(T, 1.0, x, y, &a);
    const float = @import("float.zig");
    // A[0,0] = 1 + 2*0.5 = 2
    try std.testing.expect(float.approxEqAbs(T, try a.get(0, 0), 2.0, 1e-12));
    // A[0,1] = 2 + 2*1 = 4
    try std.testing.expect(float.approxEqAbs(T, try a.get(0, 1), 4.0, 1e-12));
    // A[1,2] = 6 + 3*1.5 = 10.5
    try std.testing.expect(float.approxEqAbs(T, try a.get(1, 2), 10.5, 1e-12));
}

test "ger shape mismatch" {
    const T = f64;
    const V = Vector(T);
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 2, 3, &[_]T{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer a.deinit(std.testing.allocator);
    var x = try V.fromSlice(std.testing.allocator, &[_]T{2.0});
    defer x.deinit(std.testing.allocator);
    var y = try V.fromSlice(std.testing.allocator, &[_]T{ 0.5, 1.0, 1.5 });
    defer y.deinit(std.testing.allocator);

    try std.testing.expectError(error.ShapeMismatch, ger(T, 1.0, x, y, &a));
}

const gemm_block_size: usize = 32;

pub fn gemm(
    comptime T: type,
    trans_a: Transpose,
    trans_b: Transpose,
    alpha: T,
    a: Matrix(T),
    b: Matrix(T),
    beta: T,
    c: *Matrix(T),
) Error!void {
    _ = util.Float(T);

    const a_outer = if (trans_a == .no_trans) a.rows else a.cols;
    const a_inner = if (trans_a == .no_trans) a.cols else a.rows;
    const b_inner = if (trans_b == .no_trans) b.rows else b.cols;
    const b_outer = if (trans_b == .no_trans) b.cols else b.rows;

    if (a_inner != b_inner) return error.ShapeMismatch;
    if (c.rows != a_outer or c.cols != b_outer) return error.ShapeMismatch;

    const m = c.rows;
    const n = c.cols;
    const k = a_inner;

    // Scale C by beta
    if (beta == 0) {
        for (0..m) |i| {
            for (0..n) |j| {
                c.data[i * c.row_stride + j * c.col_stride] = 0;
            }
        }
    } else if (beta != 1) {
        for (0..m) |i| {
            for (0..n) |j| {
                c.data[i * c.row_stride + j * c.col_stride] *= beta;
            }
        }
    }

    if (alpha == 0) return;

    const nn = trans_a == .no_trans and trans_b == .no_trans;
    if (nn and m >= gemm_block_size and n >= gemm_block_size and k >= gemm_block_size) {
        gemm_blocked_nn(T, alpha, a, b, c);
    } else {
        gemm_simple(T, trans_a, trans_b, alpha, a, b, c);
    }
}

fn gemm_simple(
    comptime T: type,
    trans_a: Transpose,
    trans_b: Transpose,
    alpha: T,
    a: Matrix(T),
    b: Matrix(T),
    c: *Matrix(T),
) void {
    const m = c.rows;
    const n = c.cols;
    const k = if (trans_a == .no_trans) a.cols else a.rows;

    for (0..m) |i| {
        for (0..n) |j| {
            var sum: T = 0;
            for (0..k) |l| {
                const av = if (trans_a == .no_trans)
                    a.data[i * a.row_stride + l * a.col_stride]
                else
                    a.data[l * a.row_stride + i * a.col_stride];
                const bv = if (trans_b == .no_trans)
                    b.data[l * b.row_stride + j * b.col_stride]
                else
                    b.data[j * b.row_stride + l * b.col_stride];
                sum += av * bv;
            }
            c.data[i * c.row_stride + j * c.col_stride] += alpha * sum;
        }
    }
}

fn gemm_blocked_nn(
    comptime T: type,
    alpha: T,
    a: Matrix(T),
    b: Matrix(T),
    c: *Matrix(T),
) void {
    const m = c.rows;
    const n = c.cols;
    const k = a.cols;
    const block = gemm_block_size;

    var ii: usize = 0;
    while (ii < m) : (ii += block) {
        const i_end = @min(ii + block, m);
        var jj: usize = 0;
        while (jj < n) : (jj += block) {
            const j_end = @min(jj + block, n);
            var kk: usize = 0;
            while (kk < k) : (kk += block) {
                const k_end = @min(kk + block, k);
                var i = ii;
                while (i < i_end) : (i += 1) {
                    var j = jj;
                    while (j < j_end) : (j += 1) {
                        var sum: T = 0;
                        var l = kk;
                        while (l < k_end) : (l += 1) {
                            sum += a.data[i * a.row_stride + l * a.col_stride] *
                                b.data[l * b.row_stride + j * b.col_stride];
                        }
                        c.data[i * c.row_stride + j * c.col_stride] += alpha * sum;
                    }
                }
            }
        }
    }
}

test "gemm no_trans x no_trans f64" {
    const T = f64;
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 2, 3, &[_]T{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer a.deinit(std.testing.allocator);
    var b = try M.fromRowSlice(std.testing.allocator, 3, 2, &[_]T{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    });
    defer b.deinit(std.testing.allocator);
    var c = try M.init(std.testing.allocator, 2, 2);
    defer c.deinit(std.testing.allocator);

    try gemm(T, .no_trans, .no_trans, 1.0, a, b, 0.0, &c);
    const float = @import("float.zig");
    // C = [[22, 28], [49, 64]]
    try std.testing.expect(float.approxEqAbs(T, try c.get(0, 0), 22.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try c.get(0, 1), 28.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try c.get(1, 0), 49.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try c.get(1, 1), 64.0, 1e-12));
}

test "gemm transpose combinations f64" {
    const T = f64;
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 2, 3, &[_]T{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer a.deinit(std.testing.allocator);
    var b = try M.fromRowSlice(std.testing.allocator, 2, 3, &[_]T{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer b.deinit(std.testing.allocator);

    // trans x no_trans: A^T (3x2) * B (2x3) = 3x3
    var c_tn = try M.init(std.testing.allocator, 3, 3);
    defer c_tn.deinit(std.testing.allocator);
    try gemm(T, .trans, .no_trans, 1.0, a, b, 0.0, &c_tn);
    const float = @import("float.zig");
    try std.testing.expect(float.approxEqAbs(T, try c_tn.get(0, 0), 17.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try c_tn.get(2, 2), 45.0, 1e-12));

    // no_trans x trans: A (2x3) * B^T (3x2) = 2x2
    var c_nt = try M.init(std.testing.allocator, 2, 2);
    defer c_nt.deinit(std.testing.allocator);
    try gemm(T, .no_trans, .trans, 1.0, a, b, 0.0, &c_nt);
    try std.testing.expect(float.approxEqAbs(T, try c_nt.get(0, 0), 14.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try c_nt.get(1, 1), 77.0, 1e-12));

    // trans x trans: A^T (3x2) * B^T (3x2) -> inner dims must match: A^T is 3x2, B^T is 3x2, invalid
    // Instead use A (2x3) trans and B (3x2) no_trans? Let's use different shapes.
}

test "gemm trans x trans f64" {
    const T = f64;
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 3, 2, &[_]T{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    });
    defer a.deinit(std.testing.allocator);
    var b = try M.fromRowSlice(std.testing.allocator, 2, 3, &[_]T{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer b.deinit(std.testing.allocator);
    var c = try M.init(std.testing.allocator, 2, 2);
    defer c.deinit(std.testing.allocator);

    // A^T (2x3) * B^T (3x2) = 2x2
    try gemm(T, .trans, .trans, 1.0, a, b, 0.0, &c);
    const float = @import("float.zig");
    try std.testing.expect(float.approxEqAbs(T, try c.get(0, 0), 22.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try c.get(1, 1), 64.0, 1e-12));
}

test "gemm beta values" {
    const T = f64;
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 2, 2, &[_]T{
        1.0, 2.0,
        3.0, 4.0,
    });
    defer a.deinit(std.testing.allocator);
    var c = try M.init(std.testing.allocator, 2, 2);
    defer c.deinit(std.testing.allocator);

    try gemm(T, .no_trans, .no_trans, 1.0, a, a, 0.0, &c);
    const float = @import("float.zig");
    try std.testing.expect(float.approxEqAbs(T, try c.get(0, 0), 7.0, 1e-12));

    try gemm(T, .no_trans, .no_trans, 1.0, a, a, 1.0, &c);
    try std.testing.expect(float.approxEqAbs(T, try c.get(0, 0), 14.0, 1e-12));

    try gemm(T, .no_trans, .no_trans, 1.0, a, a, 0.5, &c);
    try std.testing.expect(float.approxEqAbs(T, try c.get(0, 0), 14.0, 1e-12));
}

test "gemm blocked path" {
    const T = f64;
    const M = Matrix(T);
    const n: usize = 40;
    var a = try M.init(std.testing.allocator, n, n);
    defer a.deinit(std.testing.allocator);
    var b = try M.init(std.testing.allocator, n, n);
    defer b.deinit(std.testing.allocator);
    var c = try M.init(std.testing.allocator, n, n);
    defer c.deinit(std.testing.allocator);

    for (0..n) |i| {
        for (0..n) |j| {
            try a.set(i, j, @as(T, @floatFromInt(i + j)));
            try b.set(i, j, @as(T, @floatFromInt(i + 2 * j)));
        }
    }

    try gemm(T, .no_trans, .no_trans, 1.0, a, b, 0.0, &c);

    // Verify a few entries against simple formula
    const float = @import("float.zig");
    try std.testing.expect(float.approxEqAbs(T, try c.get(0, 0), 20540.0, 1e-6));
    try std.testing.expect(float.approxEqAbs(T, try c.get(10, 20), 75540.0, 1e-6));
}

test "gemm shape mismatch" {
    const T = f64;
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 2, 3, &[_]T{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer a.deinit(std.testing.allocator);
    var b = try M.fromRowSlice(std.testing.allocator, 2, 2, &[_]T{
        1.0, 2.0,
        3.0, 4.0,
    });
    defer b.deinit(std.testing.allocator);
    var c = try M.init(std.testing.allocator, 2, 2);
    defer c.deinit(std.testing.allocator);

    try std.testing.expectError(error.ShapeMismatch, gemm(T, .no_trans, .no_trans, 1.0, a, b, 0.0, &c));
}

// ---------------------------------------------------------------------------
// Level-2: symmetric and triangular matrix-vector operations
// ---------------------------------------------------------------------------

pub fn symv(
    comptime T: type,
    uplo: Uplo,
    alpha: T,
    a: Matrix(T),
    x: Vector(T),
    beta: T,
    y: *Vector(T),
) Error!void {
    _ = util.Float(T);
    if (a.rows != a.cols) return error.ShapeMismatch;
    if (a.rows != x.len or a.rows != y.len) return error.ShapeMismatch;

    const n = a.rows;

    if (beta == 0) {
        for (0..n) |i| y.data[i * y.stride] = 0;
    } else if (beta != 1) {
        for (0..n) |i| y.data[i * y.stride] *= beta;
    }

    if (alpha == 0) return;

    for (0..n) |i| {
        const temp1 = alpha * x.data[i * x.stride];
        var temp2: T = 0;
        switch (uplo) {
            .upper => {
                for (0..i) |j| {
                    const a_ij = a.data[j * a.row_stride + i * a.col_stride];
                    y.data[j * y.stride] += temp1 * a_ij;
                    temp2 += a_ij * x.data[j * x.stride];
                }
                const a_ii = a.data[i * a.row_stride + i * a.col_stride];
                y.data[i * y.stride] += temp1 * a_ii + alpha * temp2;
            },
            .lower => {
                const a_ii = a.data[i * a.row_stride + i * a.col_stride];
                y.data[i * y.stride] += temp1 * a_ii;
                for (i + 1..n) |j| {
                    const a_ij = a.data[j * a.row_stride + i * a.col_stride];
                    y.data[j * y.stride] += temp1 * a_ij;
                    temp2 += a_ij * x.data[j * x.stride];
                }
                y.data[i * y.stride] += alpha * temp2;
            },
        }
    }
}

pub fn syr(
    comptime T: type,
    uplo: Uplo,
    alpha: T,
    x: Vector(T),
    a: *Matrix(T),
) Error!void {
    _ = util.Float(T);
    if (a.rows != a.cols) return error.ShapeMismatch;
    if (a.rows != x.len) return error.ShapeMismatch;
    if (alpha == 0) return;

    const n = a.rows;
    switch (uplo) {
        .upper => {
            for (0..n) |i| {
                const temp = alpha * x.data[i * x.stride];
                for (i..n) |j| {
                    a.data[i * a.row_stride + j * a.col_stride] += temp * x.data[j * x.stride];
                }
            }
        },
        .lower => {
            for (0..n) |i| {
                const temp = alpha * x.data[i * x.stride];
                for (0..i + 1) |j| {
                    a.data[i * a.row_stride + j * a.col_stride] += temp * x.data[j * x.stride];
                }
            }
        },
    }
}

pub fn syr2(
    comptime T: type,
    uplo: Uplo,
    alpha: T,
    x: Vector(T),
    y: Vector(T),
    a: *Matrix(T),
) Error!void {
    _ = util.Float(T);
    if (a.rows != a.cols) return error.ShapeMismatch;
    if (a.rows != x.len or a.rows != y.len) return error.ShapeMismatch;
    if (alpha == 0) return;

    const n = a.rows;
    switch (uplo) {
        .upper => {
            for (0..n) |i| {
                const temp1 = alpha * x.data[i * x.stride];
                const temp2 = alpha * y.data[i * y.stride];
                for (i..n) |j| {
                    a.data[i * a.row_stride + j * a.col_stride] +=
                        temp1 * y.data[j * y.stride] + temp2 * x.data[j * x.stride];
                }
            }
        },
        .lower => {
            for (0..n) |i| {
                const temp1 = alpha * x.data[i * x.stride];
                const temp2 = alpha * y.data[i * y.stride];
                for (0..i + 1) |j| {
                    a.data[i * a.row_stride + j * a.col_stride] +=
                        temp1 * y.data[j * y.stride] + temp2 * x.data[j * x.stride];
                }
            }
        },
    }
}

pub fn trmv(
    comptime T: type,
    uplo: Uplo,
    trans_a: Transpose,
    diag: Diagonal,
    a: Matrix(T),
    x: *Vector(T),
) Error!void {
    _ = util.Float(T);
    if (a.rows != a.cols) return error.ShapeMismatch;
    if (a.rows != x.len) return error.ShapeMismatch;

    const n = a.rows;
    const non_unit = diag == .non_unit;

    switch (trans_a) {
        .no_trans => {
            switch (uplo) {
                .upper => {
                    for (0..n) |ui| {
                        var temp: T = if (non_unit) a.data[ui * a.row_stride + ui * a.col_stride] * x.data[ui * x.stride] else x.data[ui * x.stride];
                        for (ui + 1..n) |j| {
                            temp += a.data[ui * a.row_stride + j * a.col_stride] * x.data[j * x.stride];
                        }
                        x.data[ui * x.stride] = temp;
                    }
                },
                .lower => {
                    var i: isize = @intCast(n - 1);
                    while (i >= 0) : (i -= 1) {
                        const ui: usize = @intCast(i);
                        var temp: T = if (non_unit) a.data[ui * a.row_stride + ui * a.col_stride] * x.data[ui * x.stride] else x.data[ui * x.stride];
                        for (0..ui) |j| {
                            temp += a.data[ui * a.row_stride + j * a.col_stride] * x.data[j * x.stride];
                        }
                        x.data[ui * x.stride] = temp;
                    }
                },
            }
        },
        .trans, .conj_trans => {
            switch (uplo) {
                .upper => {
                    var i: isize = @intCast(n - 1);
                    while (i >= 0) : (i -= 1) {
                        const ui: usize = @intCast(i);
                        const xi = x.data[ui * x.stride];
                        if (non_unit) x.data[ui * x.stride] *= a.data[ui * a.row_stride + ui * a.col_stride];
                        for (ui + 1..n) |j| {
                            x.data[j * x.stride] += a.data[ui * a.row_stride + j * a.col_stride] * xi;
                        }
                    }
                },
                .lower => {
                    for (0..n) |ui| {
                        const xi = x.data[ui * x.stride];
                        if (non_unit) x.data[ui * x.stride] *= a.data[ui * a.row_stride + ui * a.col_stride];
                        for (0..ui) |j| {
                            x.data[j * x.stride] += a.data[ui * a.row_stride + j * a.col_stride] * xi;
                        }
                    }
                },
            }
        },
    }
}

pub fn trsv(
    comptime T: type,
    uplo: Uplo,
    trans_a: Transpose,
    diag: Diagonal,
    a: Matrix(T),
    x: *Vector(T),
) Error!void {
    _ = util.Float(T);
    if (a.rows != a.cols) return error.ShapeMismatch;
    if (a.rows != x.len) return error.ShapeMismatch;

    const n = a.rows;
    const non_unit = diag == .non_unit;

    switch (trans_a) {
        .no_trans => {
            switch (uplo) {
                .upper => {
                    var i: isize = @intCast(n - 1);
                    while (i >= 0) : (i -= 1) {
                        const ui: usize = @intCast(i);
                        var temp = x.data[ui * x.stride];
                        for (ui + 1..n) |j| {
                            temp -= a.data[ui * a.row_stride + j * a.col_stride] * x.data[j * x.stride];
                        }
                        if (non_unit) temp /= a.data[ui * a.row_stride + ui * a.col_stride];
                        x.data[ui * x.stride] = temp;
                    }
                },
                .lower => {
                    for (0..n) |ui| {
                        var temp = x.data[ui * x.stride];
                        for (0..ui) |j| {
                            temp -= a.data[ui * a.row_stride + j * a.col_stride] * x.data[j * x.stride];
                        }
                        if (non_unit) temp /= a.data[ui * a.row_stride + ui * a.col_stride];
                        x.data[ui * x.stride] = temp;
                    }
                },
            }
        },
        .trans, .conj_trans => {
            switch (uplo) {
                .upper => {
                    for (0..n) |ui| {
                        var temp = x.data[ui * x.stride];
                        for (0..ui) |j| {
                            temp -= a.data[j * a.row_stride + ui * a.col_stride] * x.data[j * x.stride];
                        }
                        if (non_unit) temp /= a.data[ui * a.row_stride + ui * a.col_stride];
                        x.data[ui * x.stride] = temp;
                    }
                },
                .lower => {
                    var i: isize = @intCast(n - 1);
                    while (i >= 0) : (i -= 1) {
                        const ui: usize = @intCast(i);
                        var temp = x.data[ui * x.stride];
                        for (ui + 1..n) |j| {
                            temp -= a.data[j * a.row_stride + ui * a.col_stride] * x.data[j * x.stride];
                        }
                        if (non_unit) temp /= a.data[ui * a.row_stride + ui * a.col_stride];
                        x.data[ui * x.stride] = temp;
                    }
                },
            }
        },
    }
}

test "symv upper and lower" {
    const T = f64;
    const V = Vector(T);
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 3, 3, &[_]T{
        1.0, 2.0, 3.0,
        2.0, 4.0, 5.0,
        3.0, 5.0, 6.0,
    });
    defer a.deinit(std.testing.allocator);
    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 1.0, 1.0 });
    defer x.deinit(std.testing.allocator);
    var y = try V.init(std.testing.allocator, 3);
    defer y.deinit(std.testing.allocator);

    try symv(T, .upper, 1.0, a, x, 0.0, &y);
    const float = @import("float.zig");
    try std.testing.expect(float.approxEqAbs(T, try y.get(0), 6.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try y.get(1), 11.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try y.get(2), 14.0, 1e-12));

    try symv(T, .lower, 1.0, a, x, 0.0, &y);
    try std.testing.expect(float.approxEqAbs(T, try y.get(0), 6.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try y.get(1), 11.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try y.get(2), 14.0, 1e-12));
}

test "syr updates only requested triangle" {
    const T = f64;
    const V = Vector(T);
    const M = Matrix(T);
    var a = try M.init(std.testing.allocator, 3, 3);
    defer a.deinit(std.testing.allocator);
    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 2.0, 3.0 });
    defer x.deinit(std.testing.allocator);

    try syr(T, .upper, 1.0, x, &a);
    try std.testing.expectEqual(@as(T, 1.0), try a.get(0, 0));
    try std.testing.expectEqual(@as(T, 2.0), try a.get(0, 1));
    try std.testing.expectEqual(@as(T, 0.0), try a.get(1, 0));

    var b = try M.init(std.testing.allocator, 3, 3);
    defer b.deinit(std.testing.allocator);
    try syr(T, .lower, 1.0, x, &b);
    try std.testing.expectEqual(@as(T, 1.0), try b.get(0, 0));
    try std.testing.expectEqual(@as(T, 2.0), try b.get(1, 0));
    try std.testing.expectEqual(@as(T, 0.0), try b.get(0, 1));
}

test "syr2 basic" {
    const T = f64;
    const V = Vector(T);
    const M = Matrix(T);
    var a = try M.init(std.testing.allocator, 3, 3);
    defer a.deinit(std.testing.allocator);
    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 2.0, 3.0 });
    defer x.deinit(std.testing.allocator);
    var y = try V.fromSlice(std.testing.allocator, &[_]T{ 0.5, 1.0, 1.5 });
    defer y.deinit(std.testing.allocator);

    try syr2(T, .upper, 1.0, x, y, &a);
    const float = @import("float.zig");
    // A[i,j] = x_i*y_j + y_i*x_j
    try std.testing.expect(float.approxEqAbs(T, try a.get(0, 0), 1.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try a.get(0, 1), 2.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try a.get(1, 1), 4.0, 1e-12));
}

test "trmv unit and non_unit" {
    const T = f64;
    const V = Vector(T);
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 3, 3, &[_]T{
        1.0, 2.0, 3.0,
        0.0, 1.0, 4.0,
        0.0, 0.0, 1.0,
    });
    defer a.deinit(std.testing.allocator);

    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 1.0, 1.0 });
    defer x.deinit(std.testing.allocator);
    try trmv(T, .upper, .no_trans, .non_unit, a, &x);
    try std.testing.expectEqual(@as(T, 6.0), try x.get(0));
    try std.testing.expectEqual(@as(T, 5.0), try x.get(1));
    try std.testing.expectEqual(@as(T, 1.0), try x.get(2));

    var y = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 1.0, 1.0 });
    defer y.deinit(std.testing.allocator);
    try trmv(T, .upper, .trans, .non_unit, a, &y);
    try std.testing.expectEqual(@as(T, 1.0), try y.get(0));
    try std.testing.expectEqual(@as(T, 3.0), try y.get(1));
    try std.testing.expectEqual(@as(T, 8.0), try y.get(2));
}

test "trsv solves triangular system" {
    const T = f64;
    const V = Vector(T);
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 3, 3, &[_]T{
        2.0, 4.0, 4.0,
        0.0, 3.0, 6.0,
        0.0, 0.0, 1.0,
    });
    defer a.deinit(std.testing.allocator);
    var b = try V.fromSlice(std.testing.allocator, &[_]T{ 18.0, 15.0, 1.0 });
    defer b.deinit(std.testing.allocator);

    try trsv(T, .upper, .no_trans, .non_unit, a, &b);
    const float = @import("float.zig");
    try std.testing.expect(float.approxEqAbs(T, try b.get(0), 1.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try b.get(1), 3.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try b.get(2), 1.0, 1e-12));
}

// ---------------------------------------------------------------------------
// Level-3: symmetric and triangular matrix-matrix operations
// ---------------------------------------------------------------------------

pub fn syrk(
    comptime T: type,
    uplo: Uplo,
    trans_a: Transpose,
    alpha: T,
    a: Matrix(T),
    beta: T,
    c: *Matrix(T),
) Error!void {
    _ = util.Float(T);
    const n = c.rows;
    if (n != c.cols) return error.ShapeMismatch;

    const k = if (trans_a == .no_trans) a.cols else a.rows;
    if (trans_a == .no_trans and a.rows != n) return error.ShapeMismatch;
    if (trans_a == .no_trans and a.cols != k) return error.ShapeMismatch;
    if (trans_a != .no_trans and a.cols != n) return error.ShapeMismatch;
    if (trans_a != .no_trans and a.rows != k) return error.ShapeMismatch;

    // Scale C by beta
    if (beta == 0) {
        for (0..n) |i| {
            const start = if (uplo == .upper) i else 0;
            const end = if (uplo == .upper) n else i + 1;
            for (start..end) |j| {
                c.data[i * c.row_stride + j * c.col_stride] = 0;
            }
        }
    } else if (beta != 1) {
        for (0..n) |i| {
            const start = if (uplo == .upper) i else 0;
            const end = if (uplo == .upper) n else i + 1;
            for (start..end) |j| {
                c.data[i * c.row_stride + j * c.col_stride] *= beta;
            }
        }
    }

    if (alpha == 0) return;

    switch (trans_a) {
        .no_trans => {
            for (0..n) |i| {
                const start = if (uplo == .upper) i else 0;
                const end = if (uplo == .upper) n else i + 1;
                for (start..end) |j| {
                    var sum: T = 0;
                    for (0..k) |l| {
                        sum += a.data[i * a.row_stride + l * a.col_stride] *
                            a.data[j * a.row_stride + l * a.col_stride];
                    }
                    c.data[i * c.row_stride + j * c.col_stride] += alpha * sum;
                }
            }
        },
        .trans, .conj_trans => {
            for (0..n) |i| {
                const start = if (uplo == .upper) i else 0;
                const end = if (uplo == .upper) n else i + 1;
                for (start..end) |j| {
                    var sum: T = 0;
                    for (0..k) |l| {
                        sum += a.data[l * a.row_stride + i * a.col_stride] *
                            a.data[l * a.row_stride + j * a.col_stride];
                    }
                    c.data[i * c.row_stride + j * c.col_stride] += alpha * sum;
                }
            }
        },
    }
}

pub fn syr2k(
    comptime T: type,
    uplo: Uplo,
    trans_a: Transpose,
    alpha: T,
    a: Matrix(T),
    b: Matrix(T),
    beta: T,
    c: *Matrix(T),
) Error!void {
    _ = util.Float(T);
    const n = c.rows;
    if (n != c.cols) return error.ShapeMismatch;

    const k = if (trans_a == .no_trans) a.cols else a.rows;
    if (trans_a == .no_trans) {
        if (a.rows != n or a.cols != k) return error.ShapeMismatch;
        if (b.rows != n or b.cols != k) return error.ShapeMismatch;
    } else {
        if (a.cols != n or a.rows != k) return error.ShapeMismatch;
        if (b.cols != n or b.rows != k) return error.ShapeMismatch;
    }

    // Scale C by beta
    if (beta == 0) {
        for (0..n) |i| {
            const start = if (uplo == .upper) i else 0;
            const end = if (uplo == .upper) n else i + 1;
            for (start..end) |j| {
                c.data[i * c.row_stride + j * c.col_stride] = 0;
            }
        }
    } else if (beta != 1) {
        for (0..n) |i| {
            const start = if (uplo == .upper) i else 0;
            const end = if (uplo == .upper) n else i + 1;
            for (start..end) |j| {
                c.data[i * c.row_stride + j * c.col_stride] *= beta;
            }
        }
    }

    if (alpha == 0) return;

    switch (trans_a) {
        .no_trans => {
            for (0..n) |i| {
                const start = if (uplo == .upper) i else 0;
                const end = if (uplo == .upper) n else i + 1;
                for (start..end) |j| {
                    var sum: T = 0;
                    for (0..k) |l| {
                        sum += a.data[i * a.row_stride + l * a.col_stride] *
                            b.data[j * a.row_stride + l * a.col_stride];
                        sum += b.data[i * a.row_stride + l * a.col_stride] *
                            a.data[j * a.row_stride + l * a.col_stride];
                    }
                    c.data[i * c.row_stride + j * c.col_stride] += alpha * sum;
                }
            }
        },
        .trans, .conj_trans => {
            for (0..n) |i| {
                const start = if (uplo == .upper) i else 0;
                const end = if (uplo == .upper) n else i + 1;
                for (start..end) |j| {
                    var sum: T = 0;
                    for (0..k) |l| {
                        sum += a.data[l * a.row_stride + i * a.col_stride] *
                            b.data[l * a.row_stride + j * a.col_stride];
                        sum += b.data[l * a.row_stride + i * a.col_stride] *
                            a.data[l * a.row_stride + j * a.col_stride];
                    }
                    c.data[i * c.row_stride + j * c.col_stride] += alpha * sum;
                }
            }
        },
    }
}

pub fn trmm(
    comptime T: type,
    side: Side,
    uplo: Uplo,
    trans_a: Transpose,
    diag: Diagonal,
    alpha: T,
    a: Matrix(T),
    b: *Matrix(T),
) Error!void {
    _ = util.Float(T);
    const non_unit = diag == .non_unit;

    switch (side) {
        .left => {
            if (a.rows != a.cols) return error.ShapeMismatch;
            if (a.rows != b.rows) return error.ShapeMismatch;
            const m = b.rows;
            const n = b.cols;

            for (0..n) |j| {
                switch (uplo) {
                    .upper => {
                        switch (trans_a) {
                            .no_trans => {
                                for (0..m) |i| {
                                    var temp: T = if (non_unit) a.data[i * a.row_stride + i * a.col_stride] * b.data[i * b.row_stride + j * b.col_stride] else b.data[i * b.row_stride + j * b.col_stride];
                                    for (i + 1..m) |k| {
                                        temp += a.data[i * a.row_stride + k * a.col_stride] * b.data[k * b.row_stride + j * b.col_stride];
                                    }
                                    b.data[i * b.row_stride + j * b.col_stride] = alpha * temp;
                                }
                            },
                            .trans, .conj_trans => {
                                var i: isize = @intCast(m - 1);
                                while (i >= 0) : (i -= 1) {
                                    const ui: usize = @intCast(i);
                                    var temp: T = if (non_unit) a.data[ui * a.row_stride + ui * a.col_stride] * b.data[ui * b.row_stride + j * b.col_stride] else b.data[ui * b.row_stride + j * b.col_stride];
                                    for (0..ui) |k| {
                                        temp += a.data[k * a.row_stride + ui * a.col_stride] * b.data[k * b.row_stride + j * b.col_stride];
                                    }
                                    b.data[ui * b.row_stride + j * b.col_stride] = alpha * temp;
                                }
                            },
                        }
                    },
                    .lower => {
                        switch (trans_a) {
                            .no_trans => {
                                var i: isize = @intCast(m - 1);
                                while (i >= 0) : (i -= 1) {
                                    const ui: usize = @intCast(i);
                                    var temp: T = if (non_unit) a.data[ui * a.row_stride + ui * a.col_stride] * b.data[ui * b.row_stride + j * b.col_stride] else b.data[ui * b.row_stride + j * b.col_stride];
                                    for (0..ui) |k| {
                                        temp += a.data[ui * a.row_stride + k * a.col_stride] * b.data[k * b.row_stride + j * b.col_stride];
                                    }
                                    b.data[ui * b.row_stride + j * b.col_stride] = alpha * temp;
                                }
                            },
                            .trans, .conj_trans => {
                                for (0..m) |i| {
                                    var temp: T = if (non_unit) a.data[i * a.row_stride + i * a.col_stride] * b.data[i * b.row_stride + j * b.col_stride] else b.data[i * b.row_stride + j * b.col_stride];
                                    for (i + 1..m) |k| {
                                        temp += a.data[i * a.row_stride + k * a.col_stride] * b.data[k * b.row_stride + j * b.col_stride];
                                    }
                                    b.data[i * b.row_stride + j * b.col_stride] = alpha * temp;
                                }
                            },
                        }
                    },
                }
            }
        },
        .right => {
            if (a.rows != a.cols) return error.ShapeMismatch;
            if (a.rows != b.cols) return error.ShapeMismatch;
            const m = b.rows;
            const n = b.cols;

            for (0..m) |i| {
                switch (uplo) {
                    .upper => {
                        switch (trans_a) {
                            .no_trans => {
                                var j: isize = @intCast(n - 1);
                                while (j >= 0) : (j -= 1) {
                                    const uj: usize = @intCast(j);
                                    var temp: T = if (non_unit) a.data[uj * a.row_stride + uj * a.col_stride] * b.data[i * b.row_stride + uj * b.col_stride] else b.data[i * b.row_stride + uj * b.col_stride];
                                    for (0..uj) |k| {
                                        temp += a.data[k * a.row_stride + uj * a.col_stride] * b.data[i * b.row_stride + k * b.col_stride];
                                    }
                                    b.data[i * b.row_stride + uj * b.col_stride] = alpha * temp;
                                }
                            },
                            .trans, .conj_trans => {
                                for (0..n) |uj| {
                                    var temp: T = if (non_unit) a.data[uj * a.row_stride + uj * a.col_stride] * b.data[i * b.row_stride + uj * b.col_stride] else b.data[i * b.row_stride + uj * b.col_stride];
                                    for (uj + 1..n) |k| {
                                        temp += a.data[uj * a.row_stride + k * a.col_stride] * b.data[i * b.row_stride + k * b.col_stride];
                                    }
                                    b.data[i * b.row_stride + uj * b.col_stride] = alpha * temp;
                                }
                            },
                        }
                    },
                    .lower => {
                        switch (trans_a) {
                            .no_trans => {
                                for (0..n) |uj| {
                                    var temp: T = if (non_unit) a.data[uj * a.row_stride + uj * a.col_stride] * b.data[i * b.row_stride + uj * b.col_stride] else b.data[i * b.row_stride + uj * b.col_stride];
                                    for (uj + 1..n) |k| {
                                        temp += a.data[uj * a.row_stride + k * a.col_stride] * b.data[i * b.row_stride + k * b.col_stride];
                                    }
                                    b.data[i * b.row_stride + uj * b.col_stride] = alpha * temp;
                                }
                            },
                            .trans, .conj_trans => {
                                var j: isize = @intCast(n - 1);
                                while (j >= 0) : (j -= 1) {
                                    const uj: usize = @intCast(j);
                                    var temp: T = if (non_unit) a.data[uj * a.row_stride + uj * a.col_stride] * b.data[i * b.row_stride + uj * b.col_stride] else b.data[i * b.row_stride + uj * b.col_stride];
                                    for (0..uj) |k| {
                                        temp += a.data[k * a.row_stride + uj * a.col_stride] * b.data[i * b.row_stride + k * b.col_stride];
                                    }
                                    b.data[i * b.row_stride + uj * b.col_stride] = alpha * temp;
                                }
                            },
                        }
                    },
                }
            }
        },
    }
}

pub fn trsm(
    comptime T: type,
    side: Side,
    uplo: Uplo,
    trans_a: Transpose,
    diag: Diagonal,
    alpha: T,
    a: Matrix(T),
    b: *Matrix(T),
) Error!void {
    _ = util.Float(T);
    const non_unit = diag == .non_unit;

    // Scale B by alpha first
    if (alpha != 1) {
        for (0..b.rows) |i| {
            for (0..b.cols) |j| {
                b.data[i * b.row_stride + j * b.col_stride] *= alpha;
            }
        }
    }

    switch (side) {
        .left => {
            if (a.rows != a.cols) return error.ShapeMismatch;
            if (a.rows != b.rows) return error.ShapeMismatch;
            const m = b.rows;
            const n = b.cols;

            for (0..n) |j| {
                switch (uplo) {
                    .upper => {
                        switch (trans_a) {
                            .no_trans => {
                                var i: isize = @intCast(m - 1);
                                while (i >= 0) : (i -= 1) {
                                    const ui: usize = @intCast(i);
                                    var temp = b.data[ui * b.row_stride + j * b.col_stride];
                                    for (ui + 1..m) |k| {
                                        temp -= a.data[ui * a.row_stride + k * a.col_stride] * b.data[k * b.row_stride + j * b.col_stride];
                                    }
                                    if (non_unit) temp /= a.data[ui * a.row_stride + ui * a.col_stride];
                                    b.data[ui * b.row_stride + j * b.col_stride] = temp;
                                }
                            },
                            .trans, .conj_trans => {
                                for (0..m) |ui| {
                                    var temp = b.data[ui * b.row_stride + j * b.col_stride];
                                    for (0..ui) |k| {
                                        temp -= a.data[k * a.row_stride + ui * a.col_stride] * b.data[k * b.row_stride + j * b.col_stride];
                                    }
                                    if (non_unit) temp /= a.data[ui * a.row_stride + ui * a.col_stride];
                                    b.data[ui * b.row_stride + j * b.col_stride] = temp;
                                }
                            },
                        }
                    },
                    .lower => {
                        switch (trans_a) {
                            .no_trans => {
                                for (0..m) |ui| {
                                    var temp = b.data[ui * b.row_stride + j * b.col_stride];
                                    for (0..ui) |k| {
                                        temp -= a.data[ui * a.row_stride + k * a.col_stride] * b.data[k * b.row_stride + j * b.col_stride];
                                    }
                                    if (non_unit) temp /= a.data[ui * a.row_stride + ui * a.col_stride];
                                    b.data[ui * b.row_stride + j * b.col_stride] = temp;
                                }
                            },
                            .trans, .conj_trans => {
                                var i: isize = @intCast(m - 1);
                                while (i >= 0) : (i -= 1) {
                                    const ui: usize = @intCast(i);
                                    var temp = b.data[ui * b.row_stride + j * b.col_stride];
                                    for (ui + 1..m) |k| {
                                        temp -= a.data[k * a.row_stride + ui * a.col_stride] * b.data[k * b.row_stride + j * b.col_stride];
                                    }
                                    if (non_unit) temp /= a.data[ui * a.row_stride + ui * a.col_stride];
                                    b.data[ui * b.row_stride + j * b.col_stride] = temp;
                                }
                            },
                        }
                    },
                }
            }
        },
        .right => {
            if (a.rows != a.cols) return error.ShapeMismatch;
            if (a.rows != b.cols) return error.ShapeMismatch;
            const m = b.rows;
            const n = b.cols;

            for (0..m) |i| {
                switch (uplo) {
                    .upper => {
                        switch (trans_a) {
                            .no_trans => {
                                for (0..n) |uj| {
                                    var temp = b.data[i * b.row_stride + uj * b.col_stride];
                                    for (0..uj) |k| {
                                        temp -= a.data[k * a.row_stride + uj * a.col_stride] * b.data[i * b.row_stride + k * b.col_stride];
                                    }
                                    if (non_unit) temp /= a.data[uj * a.row_stride + uj * a.col_stride];
                                    b.data[i * b.row_stride + uj * b.col_stride] = temp;
                                }
                            },
                            .trans, .conj_trans => {
                                var j: isize = @intCast(n - 1);
                                while (j >= 0) : (j -= 1) {
                                    const uj: usize = @intCast(j);
                                    var temp = b.data[i * b.row_stride + uj * b.col_stride];
                                    for (uj + 1..n) |k| {
                                        temp -= a.data[uj * a.row_stride + k * a.col_stride] * b.data[i * b.row_stride + k * b.col_stride];
                                    }
                                    if (non_unit) temp /= a.data[uj * a.row_stride + uj * a.col_stride];
                                    b.data[i * b.row_stride + uj * b.col_stride] = temp;
                                }
                            },
                        }
                    },
                    .lower => {
                        switch (trans_a) {
                            .no_trans => {
                                var j: isize = @intCast(n - 1);
                                while (j >= 0) : (j -= 1) {
                                    const uj: usize = @intCast(j);
                                    var temp = b.data[i * b.row_stride + uj * b.col_stride];
                                    for (uj + 1..n) |k| {
                                        temp -= a.data[uj * a.row_stride + k * a.col_stride] * b.data[i * b.row_stride + k * b.col_stride];
                                    }
                                    if (non_unit) temp /= a.data[uj * a.row_stride + uj * a.col_stride];
                                    b.data[i * b.row_stride + uj * b.col_stride] = temp;
                                }
                            },
                            .trans, .conj_trans => {
                                for (0..n) |uj| {
                                    var temp = b.data[i * b.row_stride + uj * b.col_stride];
                                    for (0..uj) |k| {
                                        temp -= a.data[k * a.row_stride + uj * a.col_stride] * b.data[i * b.row_stride + k * b.col_stride];
                                    }
                                    if (non_unit) temp /= a.data[uj * a.row_stride + uj * a.col_stride];
                                    b.data[i * b.row_stride + uj * b.col_stride] = temp;
                                }
                            },
                        }
                    },
                }
            }
        },
    }
}

test "syrk upper and lower" {
    const T = f64;
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 3, 2, &[_]T{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    });
    defer a.deinit(std.testing.allocator);
    var c = try M.init(std.testing.allocator, 3, 3);
    defer c.deinit(std.testing.allocator);

    try syrk(T, .upper, .no_trans, 1.0, a, 0.0, &c);
    // C = A*A^T
    try std.testing.expectEqual(@as(T, 5.0), try c.get(0, 0));
    try std.testing.expectEqual(@as(T, 11.0), try c.get(0, 1));
    try std.testing.expectEqual(@as(T, 17.0), try c.get(0, 2));

    var c2 = try M.init(std.testing.allocator, 2, 2);
    defer c2.deinit(std.testing.allocator);
    try syrk(T, .lower, .trans, 1.0, a, 0.0, &c2);
    // C = A^T*A
    try std.testing.expectEqual(@as(T, 35.0), try c2.get(0, 0));
    try std.testing.expectEqual(@as(T, 44.0), try c2.get(1, 0));
}

test "syr2k basic" {
    const T = f64;
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 2, 2, &[_]T{
        1.0, 2.0,
        3.0, 4.0,
    });
    defer a.deinit(std.testing.allocator);
    var b = try M.fromRowSlice(std.testing.allocator, 2, 2, &[_]T{
        0.5, 1.0,
        1.5, 2.0,
    });
    defer b.deinit(std.testing.allocator);
    var c = try M.init(std.testing.allocator, 2, 2);
    defer c.deinit(std.testing.allocator);

    try syr2k(T, .upper, .no_trans, 1.0, a, b, 0.0, &c);
    // C = A*B^T + B*A^T
    try std.testing.expectEqual(@as(T, 5.0), try c.get(0, 0));
    try std.testing.expectEqual(@as(T, 11.0), try c.get(0, 1));
}

test "trmm left upper non_unit" {
    const T = f64;
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 3, 3, &[_]T{
        1.0, 2.0, 3.0,
        0.0, 1.0, 4.0,
        0.0, 0.0, 1.0,
    });
    defer a.deinit(std.testing.allocator);
    var b = try M.fromRowSlice(std.testing.allocator, 3, 2, &[_]T{
        1.0, 0.0,
        0.0, 1.0,
        0.0, 0.0,
    });
    defer b.deinit(std.testing.allocator);

    try trmm(T, .left, .upper, .no_trans, .non_unit, 1.0, a, &b);
    // First column of B should be first column of A
    try std.testing.expectEqual(@as(T, 1.0), try b.get(0, 0));
    try std.testing.expectEqual(@as(T, 2.0), try b.get(0, 1));
    try std.testing.expectEqual(@as(T, 1.0), try b.get(1, 1));
}

test "trsm left upper non_unit" {
    const T = f64;
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 3, 3, &[_]T{
        2.0, 4.0, 4.0,
        0.0, 3.0, 6.0,
        0.0, 0.0, 1.0,
    });
    defer a.deinit(std.testing.allocator);
    var b = try M.fromRowSlice(std.testing.allocator, 3, 2, &[_]T{
        18.0, 9.0,
        15.0, 12.0,
        1.0,  1.0,
    });
    defer b.deinit(std.testing.allocator);

    try trsm(T, .left, .upper, .no_trans, .non_unit, 1.0, a, &b);
    // Solution columns are [1,3,1] and [0.5,2,1]
    try std.testing.expectEqual(@as(T, 1.0), try b.get(0, 0));
    try std.testing.expectEqual(@as(T, 3.0), try b.get(1, 0));
    try std.testing.expectEqual(@as(T, 1.0), try b.get(2, 0));
    try std.testing.expectEqual(@as(T, -1.5), try b.get(0, 1));
    try std.testing.expectEqual(@as(T, 2.0), try b.get(1, 1));
    try std.testing.expectEqual(@as(T, 1.0), try b.get(2, 1));
}
