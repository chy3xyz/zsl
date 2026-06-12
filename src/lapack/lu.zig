const std = @import("std");
const blas = @import("../blas.zig");
const la = @import("../la.zig");
const util = @import("../util.zig");
const Error = @import("../errors.zig").Error;
const Matrix = la.Matrix;
const Vector = la.Vector;
const Transpose = blas.Transpose;

pub fn dgetf2(
    comptime T: type,
    m: usize,
    n: usize,
    a: *Matrix(T),
    ipiv: []usize,
) Error!bool {
    _ = util.Float(T);
    if (m == 0 or n == 0) return true;
    if (a.rows < m or a.cols < n) return error.ShapeMismatch;

    const mn = @min(m, n);
    if (ipiv.len < mn) return error.ShapeMismatch;

    var ok = true;
    for (0..mn) |j| {
        // Column j starting at row j.
        const col = Vector(T){
            .data = a.data[j * a.row_stride + j * a.col_stride ..],
            .len = m - j,
            .stride = a.row_stride,
        };
        const jp_offset = try blas.iamax(T, col);
        const jp = j + jp_offset;
        ipiv[j] = jp;

        const pivot_val = a.data[jp * a.row_stride + j * a.col_stride];
        if (pivot_val == 0) {
            ok = false;
        } else {
            if (jp != j) {
                var row_j = Vector(T){
                    .data = a.data[j * a.row_stride ..],
                    .len = n,
                    .stride = a.col_stride,
                };
                var row_jp = Vector(T){
                    .data = a.data[jp * a.row_stride ..],
                    .len = n,
                    .stride = a.col_stride,
                };
                try blas.swap(T, &row_j, &row_jp);
            }

            if (j < m - 1) {
                var col_below = Vector(T){
                    .data = a.data[(j + 1) * a.row_stride + j * a.col_stride ..],
                    .len = m - j - 1,
                    .stride = a.row_stride,
                };
                try blas.scal(T, 1.0 / pivot_val, &col_below);
            }

            if (j < mn - 1) {
                const col_below = Vector(T){
                    .data = a.data[(j + 1) * a.row_stride + j * a.col_stride ..],
                    .len = m - j - 1,
                    .stride = a.row_stride,
                };
                const row_right = Vector(T){
                    .data = a.data[j * a.row_stride + (j + 1) * a.col_stride ..],
                    .len = n - j - 1,
                    .stride = a.col_stride,
                };
                var sub = Matrix(T){
                    .data = a.data[(j + 1) * a.row_stride + (j + 1) * a.col_stride ..],
                    .rows = m - j - 1,
                    .cols = n - j - 1,
                    .row_stride = a.row_stride,
                    .col_stride = a.col_stride,
                };
                try blas.ger(T, -1.0, col_below, row_right, &sub);
            }
        }
    }
    return ok;
}

pub fn dgetrf(
    comptime T: type,
    m: usize,
    n: usize,
    a: *Matrix(T),
    ipiv: []usize,
) Error!bool {
    return dgetf2(T, m, n, a, ipiv);
}

pub fn dlaswp(
    comptime T: type,
    a: *Matrix(T),
    k1: usize,
    k2: usize,
    ipiv: []const usize,
    incx: isize,
) Error!void {
    _ = util.Float(T);
    if (k1 > k2 or k2 > ipiv.len) return error.InvalidDimension;

    var i: usize = k1;
    var ipiv_idx: usize = if (incx > 0) @intCast(incx * @as(isize, @intCast(k1))) else @intCast(incx * @as(isize, @intCast(k2 - 1)));
    const ipiv_step: usize = @intCast(@abs(incx));

    while (i < k2) : (i += 1) {
        const ip = ipiv[ipiv_idx];
        if (ip != i) {
            var row_i = Vector(T){
                .data = a.data[i * a.row_stride ..],
                .len = a.cols,
                .stride = a.col_stride,
            };
            var row_ip = Vector(T){
                .data = a.data[ip * a.row_stride ..],
                .len = a.cols,
                .stride = a.col_stride,
            };
            try blas.swap(T, &row_i, &row_ip);
        }
        if (incx > 0) {
            ipiv_idx += ipiv_step;
        } else {
            ipiv_idx -= ipiv_step;
        }
    }
}

pub fn dgetrs(
    comptime T: type,
    trans_a: Transpose,
    a: Matrix(T),
    ipiv: []const usize,
    b: *Matrix(T),
) Error!void {
    _ = util.Float(T);
    if (a.rows != a.cols) return error.ShapeMismatch;
    if (a.rows != b.rows) return error.ShapeMismatch;
    if (ipiv.len < a.rows) return error.ShapeMismatch;

    const n = a.rows;
    const nrhs = b.cols;

    if (trans_a == .no_trans) {
        try dlaswp(T, b, 0, n, ipiv, 1);

        // Forward solve L*Y = B (L unit lower triangular).
        for (0..nrhs) |j| {
            for (0..n) |k| {
                for (k + 1..n) |i| {
                    b.data[i * b.row_stride + j * b.col_stride] -=
                        a.data[i * a.row_stride + k * a.col_stride] *
                        b.data[k * b.row_stride + j * b.col_stride];
                }
            }
        }

        // Backward solve U*X = Y.
        for (0..nrhs) |j| {
            var k: isize = @intCast(n - 1);
            while (k >= 0) : (k -= 1) {
                const uk: usize = @intCast(k);
                var sum = b.data[uk * b.row_stride + j * b.col_stride];
                for (uk + 1..n) |i| {
                    sum -= a.data[uk * a.row_stride + i * a.col_stride] *
                        b.data[i * b.row_stride + j * b.col_stride];
                }
                b.data[uk * b.row_stride + j * b.col_stride] =
                    sum / a.data[uk * a.row_stride + uk * a.col_stride];
            }
        }
    } else if (trans_a == .trans or trans_a == .conj_trans) {
        // Backward solve U^T*X = B.
        for (0..nrhs) |j| {
            for (0..n) |k| {
                var sum = b.data[k * b.row_stride + j * b.col_stride];
                for (0..k) |i| {
                    sum -= a.data[i * a.row_stride + k * a.col_stride] *
                        b.data[i * b.row_stride + j * b.col_stride];
                }
                b.data[k * b.row_stride + j * b.col_stride] =
                    sum / a.data[k * a.row_stride + k * a.col_stride];
            }
        }

        // Forward solve L^T*Y = X (L unit lower triangular).
        for (0..nrhs) |j| {
            var k: isize = @intCast(n - 1);
            while (k >= 0) : (k -= 1) {
                const uk: usize = @intCast(k);
                var sum = b.data[uk * b.row_stride + j * b.col_stride];
                for (uk + 1..n) |i| {
                    sum -= a.data[i * a.row_stride + uk * a.col_stride] *
                        b.data[i * b.row_stride + j * b.col_stride];
                }
                // diagonal of L is 1, no division
                b.data[uk * b.row_stride + j * b.col_stride] = sum;
            }
        }

        try dlaswp(T, b, 0, n, ipiv, -1);
    }
}

pub fn dgesv(
    comptime T: type,
    a: *Matrix(T),
    ipiv: []usize,
    b: *Matrix(T),
) Error!bool {
    _ = util.Float(T);
    if (a.rows != a.cols) return error.ShapeMismatch;
    if (a.rows != b.rows) return error.ShapeMismatch;

    const ok = try dgetrf(T, a.rows, a.cols, a, ipiv);
    try dgetrs(T, .no_trans, a.*, ipiv, b);
    return ok;
}

fn reconstruct_lu(comptime T: type, a: Matrix(T), ipiv: []const usize, allocator: std.mem.Allocator) Error!Matrix(T) {
    const n = a.rows;
    var p = try Matrix(T).init(allocator, n, n);
    errdefer p.deinit(allocator);
    for (0..n) |i| {
        try p.set(i, i, 1.0);
    }

    // Apply swaps in reverse to get P from identity.
    var k: isize = @intCast(n - 1);
    while (k >= 0) : (k -= 1) {
        const uk: usize = @intCast(k);
        const ip = ipiv[uk];
        if (ip != uk) {
            for (0..n) |col| {
                const tmp = try p.get(uk, col);
                try p.set(uk, col, try p.get(ip, col));
                try p.set(ip, col, tmp);
            }
        }
    }

    var l = try Matrix(T).init(allocator, n, n);
    errdefer l.deinit(allocator);
    var u = try Matrix(T).init(allocator, n, n);
    errdefer u.deinit(allocator);

    for (0..n) |i| {
        for (0..n) |j| {
            if (i == j) {
                try l.set(i, j, 1.0);
                try u.set(i, j, try a.get(i, j));
            } else if (i > j) {
                try l.set(i, j, try a.get(i, j));
            } else {
                try u.set(i, j, try a.get(i, j));
            }
        }
    }

    // result = P * L * U
    var pl = try Matrix(T).init(allocator, n, n);
    errdefer pl.deinit(allocator);
    try blas.gemm(T, .no_trans, .no_trans, 1.0, p, l, 0.0, &pl);
    var result = try Matrix(T).init(allocator, n, n);
    errdefer result.deinit(allocator);
    try blas.gemm(T, .no_trans, .no_trans, 1.0, pl, u, 0.0, &result);

    p.deinit(allocator);
    l.deinit(allocator);
    u.deinit(allocator);
    pl.deinit(allocator);
    return result;
}

test "dgetf2 reconstructs original matrix" {
    const T = f64;
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 3, 3, &[_]T{
        2.0, 1.0, 1.0,
        1.0, 3.0, 2.0,
        1.0, 0.0, 0.0,
    });
    defer a.deinit(std.testing.allocator);
    const ipiv = try std.testing.allocator.alloc(usize, 3);
    defer std.testing.allocator.free(ipiv);

    const ok = try dgetf2(T, 3, 3, &a, ipiv);
    try std.testing.expect(ok);

    var reconstructed = try reconstruct_lu(T, a, ipiv, std.testing.allocator);
    defer reconstructed.deinit(std.testing.allocator);

    const original = &[_]T{
        2.0, 1.0, 1.0,
        1.0, 3.0, 2.0,
        1.0, 0.0, 0.0,
    };
    const float = @import("../float.zig");
    for (0..3) |i| {
        for (0..3) |j| {
            const expected = original[i * 3 + j];
            try std.testing.expect(float.approxEqAbs(T, try reconstructed.get(i, j), expected, 1e-10));
        }
    }
}

test "dgesv solves linear system" {
    const T = f64;
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 3, 3, &[_]T{
        2.0, 1.0, 1.0,
        1.0, 3.0, 2.0,
        1.0, 0.0, 0.0,
    });
    defer a.deinit(std.testing.allocator);
    var b = try M.fromRowSlice(std.testing.allocator, 3, 2, &[_]T{
        4.0, 1.0,
        6.0, 4.0,
        1.0, 1.0,
    });
    defer b.deinit(std.testing.allocator);
    const ipiv = try std.testing.allocator.alloc(usize, 3);
    defer std.testing.allocator.free(ipiv);

    const ok = try dgesv(T, &a, ipiv, &b);
    try std.testing.expect(ok);

    // Solution: first col [1,1,1], second col [1,5,-6]
    const float = @import("../float.zig");
    try std.testing.expect(float.approxEqAbs(T, try b.get(0, 0), 1.0, 1e-10));
    try std.testing.expect(float.approxEqAbs(T, try b.get(1, 0), 1.0, 1e-10));
    try std.testing.expect(float.approxEqAbs(T, try b.get(2, 0), 1.0, 1e-10));
    try std.testing.expect(float.approxEqAbs(T, try b.get(0, 1), 1.0, 1e-10));
    try std.testing.expect(float.approxEqAbs(T, try b.get(1, 1), 5.0, 1e-10));
    try std.testing.expect(float.approxEqAbs(T, try b.get(2, 1), -6.0, 1e-10));
}

test "dgesv detects singular matrix" {
    const T = f64;
    const M = Matrix(T);
    var a = try M.fromRowSlice(std.testing.allocator, 3, 3, &[_]T{
        1.0, 2.0, 3.0,
        2.0, 4.0, 6.0,
        1.0, 1.0, 1.0,
    });
    defer a.deinit(std.testing.allocator);
    var b = try M.fromRowSlice(std.testing.allocator, 3, 1, &[_]T{ 1.0, 2.0, 1.0 });
    defer b.deinit(std.testing.allocator);
    const ipiv = try std.testing.allocator.alloc(usize, 3);
    defer std.testing.allocator.free(ipiv);

    const ok = try dgesv(T, &a, ipiv, &b);
    try std.testing.expect(!ok);
}
