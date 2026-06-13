const std = @import("std");
const la = @import("../la.zig");
const blas = @import("../blas.zig");
const Error = @import("../errors.zig").Error;
const Data = @import("data.zig").Data;
const Stat = @import("workspace.zig").Stat;
const ParamsReg = @import("paramsreg.zig").ParamsReg;

/// `Ridge` implements linear regression with L2 regularization.
///
/// Training uses the closed-form normal equation applied to centered data:
///   theta = (X_c^T X_c + lambda I)^{-1} X_c^T y_c
/// where X_c and y_c are column-centered. The bias is recovered from the
/// original means so that it is not regularized.
pub const Ridge = struct {
    name: []const u8,
    data: *Data(f64),
    stat: *Stat(f64),
    params: *ParamsReg(f64),

    const Self = @This();

    /// Allocates a new Ridge model bound to `data`.
    pub fn init(data: *Data(f64), name: []const u8, allocator: std.mem.Allocator) Error!Self {
        if (data.nb_features == 0 or data.nb_samples == 0) return error.InvalidDimension;

        const stat = try allocator.create(Stat(f64));
        errdefer allocator.destroy(stat);
        stat.* = try Stat(f64).from_data(data.*, name, allocator);

        const params = try allocator.create(ParamsReg(f64));
        errdefer allocator.destroy(params);
        params.* = try ParamsReg(f64).init(allocator, data.nb_features);

        return .{
            .name = name,
            .data = data,
            .stat = stat,
            .params = params,
        };
    }

    /// Releases all memory owned by the model (but not the underlying `data`).
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.stat.deinit(allocator);
        allocator.destroy(self.stat);
        self.params.deinit(allocator);
        allocator.destroy(self.params);
    }

    /// Returns the model prediction for a feature vector `x`.
    pub fn predict(self: *Self, x: []const f64) f64 {
        const n = self.params.theta.len;
        const x_vec = la.Vector(f64){ .data = @constCast(x), .len = x.len, .stride = 1 };
        const theta_vec = la.Vector(f64){ .data = self.params.theta, .len = n, .stride = 1 };
        const d = blas.dot(f64, x_vec, theta_vec) catch unreachable;
        return self.params.bias + d;
    }

    /// Returns the regularized mean squared error.
    pub fn cost(self: *Self) f64 {
        const m = self.data.nb_samples;
        const n = self.data.nb_features;
        const m_f: f64 = @floatFromInt(m);
        const lambda = self.params.lambda;

        var rss: f64 = 0.0;
        const theta_vec = la.Vector(f64){ .data = self.params.theta, .len = n, .stride = 1 };
        for (0..m) |i| {
            const row = self.data.x.row(i) catch unreachable;
            const pred = self.params.bias + (blas.dot(f64, row, theta_vec) catch unreachable);
            const diff = pred - self.data.y[i];
            rss += diff * diff;
        }

        var c: f64 = (0.5 / m_f) * rss;
        var theta_sq: f64 = 0.0;
        for (self.params.theta) |tj| {
            theta_sq += tj * tj;
        }
        c += (0.5 * lambda / m_f) * theta_sq;
        return c;
    }

    /// Fits `theta` and `bias` using the closed-form Ridge solution.
    pub fn train(self: *Self, allocator: std.mem.Allocator) Error!void {
        const m = self.data.nb_samples;
        const n = self.data.nb_features;
        const m_f: f64 = @floatFromInt(m);
        const m_1: f64 = 1.0 / m_f;
        const lambda = self.params.lambda;

        self.stat.update();
        const s = self.stat.sum_x;

        var t: f64 = 0.0;
        for (self.data.y) |yi| {
            t += yi;
        }

        // r = X^T * y - (t/m) * s
        var r = try allocator.alloc(f64, n);
        errdefer allocator.free(r);
        @memset(r, 0.0);
        var r_vec = la.Vector(f64){ .data = r, .len = n, .stride = 1 };
        const y_vec = la.Vector(f64){ .data = self.data.y, .len = m, .stride = 1 };
        try blas.gemv(f64, .trans, 1.0, self.data.x, y_vec, 0.0, &r_vec);
        const tm1 = t * m_1;
        for (0..n) |j| {
            r[j] -= tm1 * s[j];
        }

        // A = X^T * X - (1/m) * outer(s, s) + lambda * I
        var A = try la.Matrix(f64).init(allocator, n, n);
        errdefer A.deinit(allocator);
        try blas.gemm(f64, .trans, .no_trans, 1.0, self.data.x, self.data.x, 0.0, &A);
        for (0..n) |i| {
            for (0..n) |j| {
                const sij = m_1 * s[i] * s[j];
                A.data[i * A.row_stride + j * A.col_stride] -= sij;
            }
        }
        for (0..n) |i| {
            A.data[i * A.row_stride + i * A.col_stride] += lambda;
        }

        // Solve the normal equations A * theta = r using LU factorization.
        var theta_vec = la.Vector(f64){
            .data = self.params.theta,
            .len = n,
            .stride = 1,
        };
        const rhs_vec = la.Vector(f64){ .data = r, .len = n, .stride = 1 };
        la.matrix_ops.solve(f64, A, rhs_vec, &theta_vec, allocator) catch |err| switch (err) {
            error.SingularMatrix => return error.InvalidDimension,
            else => return err,
        };

        allocator.free(r);
        A.deinit(allocator);

        // b = (t - dot(s, theta)) / m
        const s_vec = la.Vector(f64){ .data = s, .len = n, .stride = 1 };
        const st = try blas.dot(f64, s_vec, theta_vec);
        self.params.bias = (t - st) * m_1;
    }
};

test "Ridge recovers y = 2*x + 1 when lambda = 0" {
    const allocator = std.testing.allocator;
    const D = Data(f64);

    const xraw = &[_][]const f64{
        &[_]f64{0.0},
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
        &[_]f64{4.0},
    };
    const yraw = &[_]f64{ 1.0, 3.0, 5.0, 7.0, 9.0 };
    var data = try D.fromRawXy(allocator, xraw, yraw);
    defer data.deinit(allocator);

    var model = try Ridge.init(&data, "test_ridge", allocator);
    defer model.deinit(allocator);

    model.params.set_lambda(0.0);
    try model.train(allocator);

    try std.testing.expectApproxEqAbs(2.0, try model.params.get_theta(0), 1e-9);
    try std.testing.expectApproxEqAbs(1.0, model.params.get_bias(), 1e-9);

    const y_pred = model.predict(&[_]f64{5.0});
    try std.testing.expectApproxEqAbs(11.0, y_pred, 1e-9);
}

test "Ridge shrinks coefficients with positive lambda" {
    const allocator = std.testing.allocator;
    const D = Data(f64);

    const xraw = &[_][]const f64{
        &[_]f64{0.0},
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
        &[_]f64{4.0},
    };
    const yraw = &[_]f64{ 1.0, 3.0, 5.0, 7.0, 9.0 };
    var data = try D.fromRawXy(allocator, xraw, yraw);
    defer data.deinit(allocator);

    var model = try Ridge.init(&data, "test_ridge_shrink", allocator);
    defer model.deinit(allocator);

    model.params.set_lambda(1.0);
    try model.train(allocator);

    // Expected closed-form solution for this 1-D data with lambda=1:
    // theta = 20 / (10 + lambda) = 20 / 11, bias = (10 + 5*lambda) / (10 + lambda) = 15 / 11.
    try std.testing.expectApproxEqAbs(20.0 / 11.0, try model.params.get_theta(0), 1e-9);
    try std.testing.expectApproxEqAbs(15.0 / 11.0, model.params.get_bias(), 1e-9);
}
