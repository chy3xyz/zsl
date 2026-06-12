const std = @import("std");
const util = @import("util.zig");
const Error = @import("errors.zig").Error;
const Vector = @import("la.zig").Vector;
const Matrix = @import("la.zig").Matrix;

pub const types = @import("blas/types.zig");
pub const Transpose = types.Transpose;

test {
    _ = types;
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
    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 2.0 });
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
