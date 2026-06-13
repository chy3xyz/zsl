const std = @import("std");
const Error = @import("../errors.zig").Error;
const blas = @import("../blas.zig");
const la = @import("../la.zig");
const ComputeBackend = @import("backend.zig").ComputeBackend;

const Matrix = la.Matrix(f64);
const Vector = la.Vector(f64);

/// CPU implementation of the `ComputeBackend` interface.
///
/// This backend is stateless and delegates matrix operations to the existing
/// `zsl.blas` routines. Activation functions are implemented with simple
/// element-wise loops.
pub const CpuBackend = struct {
    const Self = @This();

    /// Return a `ComputeBackend` interface backed by this CPU implementation.
    pub fn backend(self: *Self) ComputeBackend {
        return .{
            .ptr = self,
            .vtable = &.{
                .gemm = gemmImpl,
                .gemv = gemvImpl,
                .relu = reluImpl,
                .sigmoid = sigmoidImpl,
                .tanh = tanhImpl,
                .softmax = softmaxImpl,
                .layernorm = layernormImpl,
            },
        };
    }

    fn gemmImpl(
        ptr: *anyopaque,
        allocator: std.mem.Allocator,
        a: []const f64,
        b: []const f64,
        c: []f64,
        m: usize,
        n: usize,
        k: usize,
    ) Error!void {
        _ = ptr;
        _ = allocator;

        if (a.len != m * k) return error.ShapeMismatch;
        if (b.len != k * n) return error.ShapeMismatch;
        if (c.len != m * n) return error.ShapeMismatch;

        const A = Matrix{
            .data = @constCast(a),
            .rows = m,
            .cols = k,
            .row_stride = k,
            .col_stride = 1,
        };
        const B = Matrix{
            .data = @constCast(b),
            .rows = k,
            .cols = n,
            .row_stride = n,
            .col_stride = 1,
        };
        var C = Matrix{
            .data = c,
            .rows = m,
            .cols = n,
            .row_stride = n,
            .col_stride = 1,
        };

        return blas.gemm(f64, .no_trans, .no_trans, 1.0, A, B, 0.0, &C);
    }

    fn gemvImpl(
        ptr: *anyopaque,
        allocator: std.mem.Allocator,
        a: []const f64,
        x: []const f64,
        y: []f64,
        m: usize,
        n: usize,
    ) Error!void {
        _ = ptr;
        _ = allocator;

        if (a.len != m * n) return error.ShapeMismatch;
        if (x.len != n) return error.ShapeMismatch;
        if (y.len != m) return error.ShapeMismatch;

        const A = Matrix{
            .data = @constCast(a),
            .rows = m,
            .cols = n,
            .row_stride = n,
            .col_stride = 1,
        };
        const X = Vector{
            .data = @constCast(x),
            .len = n,
            .stride = 1,
        };
        var Y = Vector{
            .data = y,
            .len = m,
            .stride = 1,
        };

        return blas.gemv(f64, .no_trans, 1.0, A, X, 0.0, &Y);
    }

    fn reluImpl(
        ptr: *anyopaque,
        allocator: std.mem.Allocator,
        x: []f64,
    ) Error!void {
        _ = ptr;
        _ = allocator;
        for (x) |*v| v.* = @max(0.0, v.*);
    }

    fn sigmoidImpl(
        ptr: *anyopaque,
        allocator: std.mem.Allocator,
        x: []f64,
    ) Error!void {
        _ = ptr;
        _ = allocator;
        for (x) |*v| v.* = 1.0 / (1.0 + std.math.exp(-v.*));
    }

    fn tanhImpl(
        ptr: *anyopaque,
        allocator: std.mem.Allocator,
        x: []f64,
    ) Error!void {
        _ = ptr;
        _ = allocator;
        for (x) |*v| v.* = std.math.tanh(v.*);
    }

    fn softmaxImpl(
        ptr: *anyopaque,
        allocator: std.mem.Allocator,
        x: []f64,
        rows: usize,
        cols: usize,
    ) Error!void {
        _ = ptr;
        _ = allocator;

        if (rows == 0 or cols == 0) return error.InvalidDimension;
        if (x.len != rows * cols) return error.ShapeMismatch;

        for (0..rows) |r| {
            const offset = r * cols;
            var max = x[offset];
            for (1..cols) |j| {
                if (x[offset + j] > max) max = x[offset + j];
            }
            var sum: f64 = 0.0;
            for (0..cols) |j| {
                const val = std.math.exp(x[offset + j] - max);
                x[offset + j] = val;
                sum += val;
            }
            if (sum == 0.0) return error.DivisionByZero;
            const inv_sum = 1.0 / sum;
            for (0..cols) |j| x[offset + j] *= inv_sum;
        }
    }

    fn layernormImpl(
        ptr: *anyopaque,
        allocator: std.mem.Allocator,
        x: []f64,
        rows: usize,
        cols: usize,
        epsilon: f64,
    ) Error!void {
        _ = ptr;
        _ = allocator;

        if (rows == 0 or cols == 0) return error.InvalidDimension;
        if (x.len != rows * cols) return error.ShapeMismatch;

        for (0..rows) |r| {
            const offset = r * cols;

            var mean: f64 = 0.0;
            for (0..cols) |j| mean += x[offset + j];
            mean /= @as(f64, @floatFromInt(cols));

            var var_sum: f64 = 0.0;
            for (0..cols) |j| {
                const d = x[offset + j] - mean;
                var_sum += d * d;
            }
            const inv_std = 1.0 / @sqrt(var_sum / @as(f64, @floatFromInt(cols)) + epsilon);

            for (0..cols) |j| x[offset + j] = (x[offset + j] - mean) * inv_std;
        }
    }
};

test "CPU backend gemm" {
    const float = @import("../float.zig");
    var cpu = CpuBackend{};
    const be = cpu.backend();

    const a = &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const b = &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var c = [_]f64{ 0.0, 0.0, 0.0, 0.0 };

    try be.gemm(std.testing.allocator, a, b, &c, 2, 2, 3);
    // A (2x3) * B (3x2) = [[22, 28], [49, 64]]
    try std.testing.expect(float.approxEqAbs(f64, c[0], 22.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, c[1], 28.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, c[2], 49.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, c[3], 64.0, 1e-12));
}

test "CPU backend gemv" {
    const float = @import("../float.zig");
    var cpu = CpuBackend{};
    const be = cpu.backend();

    const a = &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const x = &[_]f64{ 1.0, 0.5, 2.0 };
    var y = [_]f64{ 0.0, 0.0 };

    try be.gemv(std.testing.allocator, a, x, &y, 2, 3);
    try std.testing.expect(float.approxEqAbs(f64, y[0], 8.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, y[1], 18.5, 1e-12));
}

test "CPU backend activations" {
    var cpu = CpuBackend{};
    const be = cpu.backend();

    var relu_x = [_]f64{ -1.0, 0.0, 1.0 };
    try be.relu(std.testing.allocator, &relu_x);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), relu_x[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), relu_x[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), relu_x[2], 1e-12);

    var sig_x = [_]f64{0.0};
    try be.sigmoid(std.testing.allocator, &sig_x);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), sig_x[0], 1e-12);

    var tanh_x = [_]f64{1.0};
    try be.tanh(std.testing.allocator, &tanh_x);
    try std.testing.expectApproxEqAbs(std.math.tanh(@as(f64, 1.0)), tanh_x[0], 1e-12);
}

test "CPU backend softmax" {
    var cpu = CpuBackend{};
    const be = cpu.backend();

    var x = [_]f64{ 1.0, 2.0, 3.0 };
    try be.softmax(std.testing.allocator, &x, 1, 3);

    var sum: f64 = 0.0;
    for (x) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-12);
}

test "CPU backend layernorm" {
    var cpu = CpuBackend{};
    const be = cpu.backend();

    var x = [_]f64{ 1.0, 2.0, 3.0 };
    try be.layernorm(std.testing.allocator, &x, 1, 3, 1e-8);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), x[1], 1e-12);
}

test "CPU backend rejects shape mismatches" {
    var cpu = CpuBackend{};
    const be = cpu.backend();

    const a = &[_]f64{1.0};
    const b = &[_]f64{2.0};
    var c = [_]f64{0.0};

    try std.testing.expectError(error.ShapeMismatch, be.gemm(std.testing.allocator, a, b, &c, 2, 2, 2));
    try std.testing.expectError(error.ShapeMismatch, be.gemv(std.testing.allocator, a, b, &c, 2, 2));
    try std.testing.expectError(error.ShapeMismatch, be.softmax(std.testing.allocator, &c, 2, 2));
    try std.testing.expectError(error.ShapeMismatch, be.layernorm(std.testing.allocator, &c, 2, 2, 1e-8));
}
