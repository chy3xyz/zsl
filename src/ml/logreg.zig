const std = @import("std");
const Error = @import("../errors.zig").Error;
const Data = @import("data.zig").Data;
const Stat = @import("workspace.zig").Stat;
const ParamsReg = @import("paramsreg.zig").ParamsReg;

/// `LogReg` implements binary logistic regression with L2 regularization.
///
/// The model stores a reference to the training data, per-column statistics,
/// regression parameters, and a small workspace used by `cost`, `gradients`,
/// and `train`.
pub const LogReg = struct {
    name: []const u8,
    data: *Data(f64),
    params: *ParamsReg(f64),
    stat: *Stat(f64),
    ybar: []f64,
    l: []f64,
    hmy: []f64,

    const Self = @This();

    /// Allocates and initializes a `LogReg` model for the supplied data.
    ///
    /// `name` is stored as a slice; the caller must ensure it outlives the model.
    pub fn init(data: *Data(f64), name: []const u8, allocator: std.mem.Allocator) Error!Self {
        if (data.y.len == 0 or data.nb_features == 0) return error.InvalidDimension;

        const stat = try allocator.create(Stat(f64));
        errdefer allocator.destroy(stat);
        stat.* = try Stat(f64).from_data(data.*, name, allocator);
        errdefer stat.deinit(allocator);

        const params = try allocator.create(ParamsReg(f64));
        errdefer allocator.destroy(params);
        params.* = try ParamsReg(f64).init(allocator, data.nb_features);
        errdefer params.deinit(allocator);

        const ybar = try allocator.alloc(f64, data.nb_samples);
        errdefer allocator.free(ybar);
        const l = try allocator.alloc(f64, data.nb_samples);
        errdefer allocator.free(l);
        const hmy = try allocator.alloc(f64, data.nb_samples);
        errdefer allocator.free(hmy);

        @memset(ybar, 0);
        @memset(l, 0);
        @memset(hmy, 0);

        var self = Self{
            .name = name,
            .data = data,
            .params = params,
            .stat = stat,
            .ybar = ybar,
            .l = l,
            .hmy = hmy,
        };
        self.updateYbar();
        return self;
    }

    /// Releases all memory owned by the model.
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.stat.deinit(allocator);
        allocator.destroy(self.stat);
        self.params.deinit(allocator);
        allocator.destroy(self.params);
        allocator.free(self.ybar);
        allocator.free(self.l);
        allocator.free(self.hmy);
        self.stat = undefined;
        self.params = undefined;
        self.ybar = &[_]f64{};
        self.l = &[_]f64{};
        self.hmy = &[_]f64{};
    }

    fn updateYbar(self: *Self) void {
        const m = self.data.nb_samples;
        if (m == 0) return;
        const m_1 = 1.0 / @as(f64, @floatFromInt(m));
        for (0..m) |i| {
            self.ybar[i] = (1.0 - self.data.y[i]) * m_1;
        }
    }

    /// Numerically stable sigmoid function.
    pub fn sigmoid(z: f64) f64 {
        if (z >= 0.0) {
            const ez = std.math.exp(-z);
            return 1.0 / (1.0 + ez);
        } else {
            const ez = std.math.exp(z);
            return ez / (1.0 + ez);
        }
    }

    /// Returns the model's predicted probability for a single observation.
    pub fn predict(self: *Self, x: []const f64) f64 {
        std.debug.assert(x.len == self.data.nb_features);
        const theta = self.params.theta;
        var sum: f64 = self.params.bias;
        for (0..x.len) |i| {
            sum += x[i] * theta[i];
        }
        return sigmoid(sum);
    }

    fn safeLog1pExp(z: f64) f64 {
        if (z < -500.0) return -z;
        return std.math.log1p(std.math.exp(-z));
    }

    /// Computes the logistic loss plus L2 regularization.
    pub fn cost(self: *Self) f64 {
        const m = self.data.nb_samples;
        const m_1 = 1.0 / @as(f64, @floatFromInt(m));
        const lambda = self.params.lambda;
        const theta = self.params.theta;

        self.calcl();

        var sq: f64 = 0.0;
        for (0..m) |i| {
            sq += safeLog1pExp(self.l[i]);
        }
        var c: f64 = sq * m_1;
        for (0..m) |i| {
            c += self.ybar[i] * self.l[i];
        }
        if (lambda > 0.0) {
            var theta_norm_sq: f64 = 0.0;
            for (theta) |t| {
                theta_norm_sq += t * t;
            }
            c += 0.5 * lambda * m_1 * theta_norm_sq;
        }
        return c;
    }

    /// Computes the gradient of the cost with respect to `theta`.
    ///
    /// The returned slice is owned by the caller.
    pub fn gradients(self: *Self, allocator: std.mem.Allocator) Error![]f64 {
        const m = self.data.nb_samples;
        const n = self.data.nb_features;
        const m_1 = 1.0 / @as(f64, @floatFromInt(m));
        const lambda = self.params.lambda;
        const theta = self.params.theta;

        self.calcl();
        self.calchmy();

        const grad = try allocator.alloc(f64, n);
        errdefer allocator.free(grad);
        @memset(grad, 0.0);

        for (0..m) |i| {
            const hmy_i = self.hmy[i];
            for (0..n) |j| {
                const x_ij = try self.data.x.get(i, j);
                grad[j] += x_ij * hmy_i;
            }
        }
        for (0..n) |j| {
            grad[j] *= m_1;
            if (lambda > 0.0) {
                grad[j] += lambda * m_1 * theta[j];
            }
        }
        return grad;
    }

    /// Trains the model using batch gradient descent.
    pub fn train(self: *Self, allocator: std.mem.Allocator, epochs: usize, alpha: f64) Error!void {
        const m = self.data.nb_samples;
        const m_1 = 1.0 / @as(f64, @floatFromInt(m));

        for (0..epochs) |_| {
            const grad = try self.gradients(allocator);
            defer allocator.free(grad);

            for (0..grad.len) |j| {
                self.params.theta[j] -= alpha * grad[j];
            }

            var grad_b: f64 = 0.0;
            for (0..m) |i| {
                grad_b += self.hmy[i];
            }
            self.params.bias -= alpha * grad_b * m_1;
        }
    }

    /// Computes `l[i] = bias + dot(row_i, theta)` for every sample.
    fn calcl(self: *Self) void {
        const m = self.data.nb_samples;
        const n = self.data.nb_features;
        const bias = self.params.bias;
        const theta = self.params.theta;
        for (0..m) |i| {
            const row = self.data.x.row(i) catch unreachable;
            var sum: f64 = bias;
            for (0..n) |j| {
                sum += row.data[j * row.stride] * theta[j];
            }
            self.l[i] = sum;
        }
    }

    /// Computes `hmy[i] = sigmoid(l[i]) - y[i]` for every sample.
    fn calchmy(self: *Self) void {
        const m = self.data.nb_samples;
        for (0..m) |i| {
            self.hmy[i] = sigmoid(self.l[i]) - self.data.y[i];
        }
    }
};

test "LogReg trains on a tiny linearly separable dataset" {
    const allocator = std.testing.allocator;
    const D = Data(f64);

    // One-dimensional data: negative inputs map to class 0, positive to class 1.
    const xraw = &[_][]const f64{
        &[_]f64{-1.0},
        &[_]f64{-0.5},
        &[_]f64{0.5},
        &[_]f64{1.0},
    };
    const yraw = &[_]f64{ 0.0, 0.0, 1.0, 1.0 };

    var data = try D.fromRawXy(allocator, xraw, yraw);
    defer data.deinit(allocator);

    var model = try LogReg.init(&data, "logreg_test", allocator);
    defer model.deinit(allocator);

    // Sanity-check initial cost.
    const initial_cost = model.cost();
    try std.testing.expect(std.math.isFinite(initial_cost));

    // Train with a moderate learning rate.
    try model.train(allocator, 2000, 0.5);

    // After training, the model should separate the classes confidently.
    const p_neg = model.predict(&[_]f64{-1.0});
    const p_pos = model.predict(&[_]f64{1.0});
    try std.testing.expect(p_neg < 0.5);
    try std.testing.expect(p_pos > 0.5);

    // Cost should have decreased.
    const final_cost = model.cost();
    try std.testing.expect(final_cost < initial_cost);
}
