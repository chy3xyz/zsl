const std = @import("std");
const la = @import("../la.zig");
const blas = @import("../blas.zig");
const Error = @import("../errors.zig").Error;
const Data = @import("data.zig").Data;
const Stat = @import("workspace.zig").Stat;
const ParamsReg = @import("paramsreg.zig").ParamsReg;

/// `LinReg` implements ordinary linear regression with an optional L2 penalty.
pub const LinReg = struct {
    name: []const u8,
    data: *Data(f64),
    stat: *Stat(f64),
    params: *ParamsReg(f64),
    e: []f64,

    const Self = @This();

    /// Allocates a new linear regression model bound to `data`.
    pub fn init(data: *Data(f64), name: []const u8, allocator: std.mem.Allocator) Error!Self {
        if (data.nb_features == 0 or data.nb_samples == 0) return error.InvalidDimension;

        const stat = try allocator.create(Stat(f64));
        errdefer allocator.destroy(stat);
        stat.* = try Stat(f64).from_data(data.*, name, allocator);

        const params = try allocator.create(ParamsReg(f64));
        errdefer allocator.destroy(params);
        params.* = try ParamsReg(f64).init(allocator, data.nb_features);

        const e = try allocator.alloc(f64, data.nb_samples);
        errdefer allocator.free(e);
        @memset(e, 0);

        return .{
            .name = name,
            .data = data,
            .stat = stat,
            .params = params,
            .e = e,
        };
    }

    /// Releases all memory owned by the model (but not the underlying `data`).
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.stat.deinit(allocator);
        allocator.destroy(self.stat);
        self.params.deinit(allocator);
        allocator.destroy(self.params);
        allocator.free(self.e);
        self.e = &[_]f64{};
    }

    /// Computes `e = b*o + X*theta - y` into `self.e`.
    fn computeError(self: *Self) Error!void {
        const m = self.data.nb_samples;
        const n = self.data.nb_features;

        @memset(self.e, self.params.bias);
        var e_vec = la.Vector(f64){ .data = self.e, .len = m, .stride = 1 };
        const theta_vec = la.Vector(f64){ .data = self.params.theta, .len = n, .stride = 1 };
        try blas.gemv(f64, .no_trans, 1.0, self.data.x, theta_vec, 1.0, &e_vec);
        for (0..m) |i| {
            self.e[i] -= self.data.y[i];
        }
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
        self.computeError() catch unreachable;

        const m_f: f64 = @floatFromInt(self.data.nb_samples);
        var sum_sq: f64 = 0.0;
        for (self.e) |ei| {
            sum_sq += ei * ei;
        }
        var c: f64 = (0.5 / m_f) * sum_sq;

        const lambda = self.params.lambda;
        if (lambda > 0) {
            var theta_sq: f64 = 0.0;
            for (self.params.theta) |tj| {
                theta_sq += tj * tj;
            }
            c += (0.5 * lambda / m_f) * theta_sq;
        }
        return c;
    }

    /// Returns the gradient of the cost w.r.t. theta and the bias.
    /// The caller owns `dcdtheta` and must free it with `allocator`.
    pub fn gradients(self: *Self, allocator: std.mem.Allocator) Error!struct { dcdtheta: []f64, dcdb: f64 } {
        const m = self.data.nb_samples;
        const n = self.data.nb_features;
        const m_f: f64 = @floatFromInt(m);
        const m_1: f64 = 1.0 / m_f;
        const lambda = self.params.lambda;

        try self.computeError();

        var dcdtheta = try allocator.alloc(f64, n);
        errdefer allocator.free(dcdtheta);
        @memset(dcdtheta, 0.0);
        var dcdtheta_vec = la.Vector(f64){ .data = dcdtheta, .len = n, .stride = 1 };
        const e_vec = la.Vector(f64){ .data = self.e, .len = m, .stride = 1 };
        try blas.gemv(f64, .trans, m_1, self.data.x, e_vec, 0.0, &dcdtheta_vec);

        if (lambda > 0) {
            for (0..n) |j| {
                dcdtheta[j] += (lambda * m_1) * self.params.theta[j];
            }
        }

        var dcdb: f64 = 0.0;
        for (self.e) |ei| {
            dcdb += ei;
        }
        dcdb *= m_1;

        return .{ .dcdtheta = dcdtheta, .dcdb = dcdb };
    }

    /// Fits `theta` and `bias` using the closed-form normal equations.
    pub fn train(self: *Self, allocator: std.mem.Allocator) Error!void {
        const m = self.data.nb_samples;
        const n = self.data.nb_features;
        const m_f: f64 = @floatFromInt(m);
        const m_1: f64 = 1.0 / m_f;

        self.stat.update();
        const s = self.stat.sum_x;

        var t: f64 = 0.0;
        for (self.data.y) |yi| {
            t += yi;
        }

        // a = X^T * y (reused later as r)
        var a = try allocator.alloc(f64, n);
        errdefer allocator.free(a);
        @memset(a, 0.0);
        var a_vec = la.Vector(f64){ .data = a, .len = n, .stride = 1 };
        const y_vec = la.Vector(f64){ .data = self.data.y, .len = m, .stride = 1 };
        try blas.gemv(f64, .trans, 1.0, self.data.x, y_vec, 0.0, &a_vec);

        // r = a - (t/m) * s
        const tm1 = t * m_1;
        for (0..n) |j| {
            a[j] -= tm1 * s[j];
        }

        // A = X^T * X
        var A = try la.Matrix(f64).init(allocator, n, n);
        errdefer A.deinit(allocator);
        try blas.gemm(f64, .trans, .no_trans, 1.0, self.data.x, self.data.x, 0.0, &A);

        // A -= (1/m) * outer(s, s)
        for (0..n) |i| {
            for (0..n) |j| {
                const sij = m_1 * s[i] * s[j];
                A.data[i * A.row_stride + j * A.col_stride] -= sij;
            }
        }

        // A += lambda * I
        const lambda = self.params.lambda;
        for (0..n) |i| {
            A.data[i * A.row_stride + i * A.col_stride] += lambda;
        }

        // Solve A * theta = r
        var theta_vec = la.Vector(f64){
            .data = self.params.theta,
            .len = n,
            .stride = 1,
        };
        const r_vec = la.Vector(f64){ .data = a, .len = n, .stride = 1 };
        try la.matrix_ops.solve(f64, A, r_vec, &theta_vec, allocator);

        allocator.free(a);
        A.deinit(allocator);

        // b = (t - dot(s, theta)) / m
        const s_vec = la.Vector(f64){ .data = s, .len = n, .stride = 1 };
        const st = try blas.dot(f64, s_vec, theta_vec);
        self.params.bias = (t - st) * m_1;
    }
};

test "LinReg trains and predicts y = 2*x + 1" {
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

    var reg = try LinReg.init(&data, "test_linreg", allocator);
    defer reg.deinit(allocator);

    try reg.train(allocator);

    try std.testing.expectApproxEqAbs(2.0, try reg.params.get_theta(0), 1e-9);
    try std.testing.expectApproxEqAbs(1.0, reg.params.get_bias(), 1e-9);

    const y_pred = reg.predict(&[_]f64{5.0});
    try std.testing.expectApproxEqAbs(11.0, y_pred, 1e-9);
}
