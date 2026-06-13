const std = @import("std");
const la = @import("../la.zig");
const blas = @import("../blas.zig");
const Error = @import("../errors.zig").Error;
const Data = @import("data.zig").Data;
const Stat = @import("workspace.zig").Stat;
const ParamsReg = @import("paramsreg.zig").ParamsReg;

/// `Lasso` implements linear regression with L1 regularization.
///
/// Training uses coordinate descent with soft-thresholding.
pub const Lasso = struct {
    name: []const u8,
    data: *Data(f64),
    params: *ParamsReg(f64),
    stat: *Stat(f64),

    const Self = @This();

    /// Allocates a new Lasso model bound to `data`.
    pub fn init(data: *Data(f64), name: []const u8, allocator: std.mem.Allocator) Error!Self {
        if (data.nb_samples == 0 or data.nb_features == 0) return error.InvalidDimension;

        const stat = try allocator.create(Stat(f64));
        errdefer allocator.destroy(stat);
        stat.* = try Stat(f64).from_data(data.*, name, allocator);
        errdefer stat.deinit(allocator);

        const params = try allocator.create(ParamsReg(f64));
        errdefer allocator.destroy(params);
        params.* = try ParamsReg(f64).init(allocator, data.nb_features);
        errdefer params.deinit(allocator);

        return .{
            .name = name,
            .data = data,
            .params = params,
            .stat = stat,
        };
    }

    /// Releases all memory owned by the model (but not the underlying `data`).
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.stat.deinit(allocator);
        allocator.destroy(self.stat);
        self.params.deinit(allocator);
        allocator.destroy(self.params);
    }

    /// Returns the prediction for a feature vector `x`.
    pub fn predict(self: *Self, x: []const f64) f64 {
        const n = self.params.theta.len;
        if (x.len != n) unreachable;

        const x_vec = la.Vector(f64){
            .data = @constCast(x),
            .len = x.len,
            .stride = 1,
        };
        const theta_vec = la.Vector(f64){
            .data = self.params.theta,
            .len = n,
            .stride = 1,
        };
        const d = blas.dot(f64, x_vec, theta_vec) catch unreachable;
        return self.params.bias + d;
    }

    /// Returns the regularized cost: MSE + L1 penalty.
    pub fn cost(self: *Self) f64 {
        const m = self.data.nb_samples;
        const n = self.data.nb_features;
        const m_f: f64 = @floatFromInt(m);

        const theta_vec = la.Vector(f64){
            .data = self.params.theta,
            .len = n,
            .stride = 1,
        };

        var rss: f64 = 0.0;
        for (0..m) |i| {
            const row = self.data.x.row(i) catch unreachable;
            const pred = self.params.bias + (blas.dot(f64, row, theta_vec) catch unreachable);
            const diff = pred - self.data.y[i];
            rss += diff * diff;
        }

        var c: f64 = (1.0 / m_f) * rss;
        for (self.params.theta) |tj| {
            c += self.params.lambda * @abs(tj);
        }
        return c;
    }

    /// Fits `theta` and `bias` using coordinate descent.
    pub fn train(self: *Self, allocator: std.mem.Allocator, epochs: usize, lambda: f64) Error!void {
        const m = self.data.nb_samples;
        const n = self.data.nb_features;
        if (m == 0 or n == 0) return error.InvalidDimension;

        self.params.lambda = lambda;
        self.stat.update();

        const m_f: f64 = @floatFromInt(m);

        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const train_allocator = arena.allocator();

        const residual = try train_allocator.alloc(f64, m);
        const x_norms = try train_allocator.alloc(f64, n);

        // Initialize theta to zero and bias to the mean of y.
        @memset(self.params.theta, 0.0);
        var y_mean: f64 = 0.0;
        for (self.data.y) |yi| {
            y_mean += yi;
        }
        y_mean /= m_f;
        self.params.bias = y_mean;

        for (0..m) |i| {
            residual[i] = self.data.y[i] - y_mean;
        }

        // Compute per-feature squared norms.
        for (0..n) |j| {
            const col = try self.data.x.col(j);
            x_norms[j] = try blas.dot(f64, col, col);
        }

        const threshold = lambda * m_f / 2.0;

        for (0..epochs) |_| {
            for (0..n) |j| {
                const col = try self.data.x.col(j);
                const old_coef = self.params.theta[j];

                // Remove the current coefficient's contribution from the residual.
                if (old_coef != 0.0) {
                    for (0..m) |i| {
                        residual[i] += col.data[i * col.stride] * old_coef;
                    }
                }

                // Compute the partial regression coefficient.
                var rho: f64 = 0.0;
                for (0..m) |i| {
                    rho += col.data[i * col.stride] * residual[i];
                }

                // Apply soft-thresholding (L1 regularization).
                if (x_norms[j] > 0.0) {
                    self.params.theta[j] = softThreshold(rho, threshold) / x_norms[j];
                } else {
                    self.params.theta[j] = 0.0;
                }

                // Add the new coefficient's contribution back to the residual.
                const new_coef = self.params.theta[j];
                if (new_coef != 0.0) {
                    for (0..m) |i| {
                        residual[i] -= col.data[i * col.stride] * new_coef;
                    }
                }

                // Update the bias to keep the residual mean zero.
                var r_mean: f64 = 0.0;
                for (residual) |ri| {
                    r_mean += ri;
                }
                r_mean /= m_f;
                self.params.bias += r_mean;
                for (residual) |*ri| {
                    ri.* -= r_mean;
                }
            }
        }
    }
};

fn softThreshold(x: f64, lambda: f64) f64 {
    if (x > lambda) {
        return x - lambda;
    } else if (x < -lambda) {
        return x + lambda;
    }
    return 0.0;
}

test {
    const allocator = std.testing.allocator;
    const D = Data(f64);

    const xraw = &[_][]const f64{
        &[_]f64{0.0},
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
        &[_]f64{4.0},
    };
    // y = 2*x + 1 with a small amount of noise.
    const yraw = &[_]f64{ 1.001, 2.998, 5.002, 6.999, 9.001 };

    var data = try D.fromRawXy(allocator, xraw, yraw);
    defer data.deinit(allocator);

    var model = try Lasso.init(&data, "test_lasso", allocator);
    defer model.deinit(allocator);

    try model.train(allocator, 2000, 0.001);

    try std.testing.expectApproxEqAbs(2.0, try model.params.get_theta(0), 1e-3);
    try std.testing.expectApproxEqAbs(1.0, model.params.get_bias(), 1e-3);

    const y_pred = model.predict(&[_]f64{5.0});
    try std.testing.expectApproxEqAbs(11.0, y_pred, 1e-3);
}
