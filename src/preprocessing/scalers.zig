const std = @import("std");
const Error = @import("../errors.zig").Error;
const Matrix = @import("../la.zig").Matrix(f64);

/// StandardScaler standardizes features by removing the mean and scaling to
/// unit variance: z = (x - mean) / std.
pub const StandardScaler = struct {
    allocator: std.mem.Allocator,
    fitted: bool,
    mean_: []f64,
    std_: []f64,
    n_features: usize,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .fitted = false,
            .mean_ = &[_]f64{},
            .std_ = &[_]f64{},
            .n_features = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.mean_);
        self.allocator.free(self.std_);
        self.mean_ = &[_]f64{};
        self.std_ = &[_]f64{};
        self.fitted = false;
        self.n_features = 0;
    }

    pub fn fit(self: *Self, x: Matrix) Error!void {
        if (x.rows == 0 or x.cols == 0) return error.InvalidDimension;

        self.deinit();
        self.n_features = x.cols;
        self.mean_ = try self.allocator.alloc(f64, self.n_features);
        errdefer self.allocator.free(self.mean_);
        self.std_ = try self.allocator.alloc(f64, self.n_features);
        errdefer self.allocator.free(self.std_);
        @memset(self.mean_, 0);
        @memset(self.std_, 0);

        // Compute mean for each feature.
        for (0..self.n_features) |j| {
            var sum: f64 = 0.0;
            for (0..x.rows) |i| {
                sum += try x.get(i, j);
            }
            self.mean_[j] = sum / @as(f64, @floatFromInt(x.rows));
        }

        // Compute population standard deviation for each feature.
        for (0..self.n_features) |j| {
            var sum_sq: f64 = 0.0;
            for (0..x.rows) |i| {
                const diff = (try x.get(i, j)) - self.mean_[j];
                sum_sq += diff * diff;
            }
            const std_dev = std.math.sqrt(sum_sq / @as(f64, @floatFromInt(x.rows)));
            self.std_[j] = if (std_dev == 0.0) 1.0 else std_dev;
        }

        self.fitted = true;
    }

    pub fn transform(self: *Self, x: Matrix) Error!Matrix {
        if (!self.fitted) return error.NotFitted;
        if (x.cols != self.n_features) return error.ShapeMismatch;

        var result = try Matrix.init(self.allocator, x.rows, x.cols);
        errdefer result.deinit(self.allocator);
        for (0..x.rows) |i| {
            for (0..x.cols) |j| {
                const v = try x.get(i, j);
                try result.set(i, j, (v - self.mean_[j]) / self.std_[j]);
            }
        }
        return result;
    }

    pub fn fit_transform(self: *Self, x: Matrix) Error!Matrix {
        try self.fit(x);
        return try self.transform(x);
    }

    pub fn inverse_transform(self: *Self, x: Matrix) Error!Matrix {
        if (!self.fitted) return error.NotFitted;
        if (x.cols != self.n_features) return error.ShapeMismatch;

        var result = try Matrix.init(self.allocator, x.rows, x.cols);
        errdefer result.deinit(self.allocator);
        for (0..x.rows) |i| {
            for (0..x.cols) |j| {
                const v = try x.get(i, j);
                try result.set(i, j, v * self.std_[j] + self.mean_[j]);
            }
        }
        return result;
    }
};

/// MinMaxScaler transforms features by scaling each feature to the configured
/// range [feature_min, feature_max].
pub const MinMaxScaler = struct {
    allocator: std.mem.Allocator,
    fitted: bool,
    min_: []f64,
    max_: []f64,
    data_range_: []f64,
    n_features: usize,
    feature_min: f64,
    feature_max: f64,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .fitted = false,
            .min_ = &[_]f64{},
            .max_ = &[_]f64{},
            .data_range_ = &[_]f64{},
            .n_features = 0,
            .feature_min = 0.0,
            .feature_max = 1.0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.min_);
        self.allocator.free(self.max_);
        self.allocator.free(self.data_range_);
        self.min_ = &[_]f64{};
        self.max_ = &[_]f64{};
        self.data_range_ = &[_]f64{};
        self.fitted = false;
        self.n_features = 0;
    }

    pub fn fit(self: *Self, x: Matrix) Error!void {
        if (x.rows == 0 or x.cols == 0) return error.InvalidDimension;

        self.deinit();
        self.n_features = x.cols;
        self.min_ = try self.allocator.alloc(f64, self.n_features);
        errdefer self.allocator.free(self.min_);
        self.max_ = try self.allocator.alloc(f64, self.n_features);
        errdefer self.allocator.free(self.max_);
        self.data_range_ = try self.allocator.alloc(f64, self.n_features);
        errdefer self.allocator.free(self.data_range_);
        @memset(self.min_, std.math.inf(f64));
        @memset(self.max_, -std.math.inf(f64));
        @memset(self.data_range_, 0);

        for (0..self.n_features) |j| {
            for (0..x.rows) |i| {
                const v = try x.get(i, j);
                if (v < self.min_[j]) self.min_[j] = v;
                if (v > self.max_[j]) self.max_[j] = v;
            }
            const range = self.max_[j] - self.min_[j];
            self.data_range_[j] = if (range == 0.0) 1.0 else range;
        }

        self.fitted = true;
    }

    pub fn transform(self: *Self, x: Matrix) Error!Matrix {
        if (!self.fitted) return error.NotFitted;
        if (x.cols != self.n_features) return error.ShapeMismatch;

        const feature_range = self.feature_max - self.feature_min;
        var result = try Matrix.init(self.allocator, x.rows, x.cols);
        errdefer result.deinit(self.allocator);
        for (0..x.rows) |i| {
            for (0..x.cols) |j| {
                const v = try x.get(i, j);
                const scaled = (v - self.min_[j]) / self.data_range_[j];
                try result.set(i, j, scaled * feature_range + self.feature_min);
            }
        }
        return result;
    }

    pub fn fit_transform(self: *Self, x: Matrix) Error!Matrix {
        try self.fit(x);
        return try self.transform(x);
    }

    pub fn inverse_transform(self: *Self, x: Matrix) Error!Matrix {
        if (!self.fitted) return error.NotFitted;
        if (x.cols != self.n_features) return error.ShapeMismatch;

        const feature_range = self.feature_max - self.feature_min;
        var result = try Matrix.init(self.allocator, x.rows, x.cols);
        errdefer result.deinit(self.allocator);
        for (0..x.rows) |i| {
            for (0..x.cols) |j| {
                const v = try x.get(i, j);
                const unscaled = (v - self.feature_min) / feature_range;
                try result.set(i, j, unscaled * self.data_range_[j] + self.min_[j]);
            }
        }
        return result;
    }
};

test "StandardScaler fit and transform" {
    const M = Matrix;
    var scaler = StandardScaler.init(std.testing.allocator);
    defer scaler.deinit();

    var x = try M.fromRowSlice(std.testing.allocator, 2, 2, &[_]f64{
        0.0, 0.0,
        2.0, 4.0,
    });
    defer x.deinit(std.testing.allocator);

    try scaler.fit(x);
    try std.testing.expectEqual(2, scaler.n_features);
    try std.testing.expectApproxEqAbs(1.0, scaler.mean_[0], 1e-12);
    try std.testing.expectApproxEqAbs(2.0, scaler.mean_[1], 1e-12);

    var z = try scaler.transform(x);
    defer z.deinit(std.testing.allocator);
    try std.testing.expectApproxEqAbs(-1.0, try z.get(0, 0), 1e-12);
    try std.testing.expectApproxEqAbs(-1.0, try z.get(0, 1), 1e-12);
    try std.testing.expectApproxEqAbs(1.0, try z.get(1, 0), 1e-12);
    try std.testing.expectApproxEqAbs(1.0, try z.get(1, 1), 1e-12);
}

test "StandardScaler inverse_transform" {
    const M = Matrix;
    var scaler = StandardScaler.init(std.testing.allocator);
    defer scaler.deinit();

    var x = try M.fromRowSlice(std.testing.allocator, 2, 2, &[_]f64{
        1.0, 2.0,
        3.0, 4.0,
    });
    defer x.deinit(std.testing.allocator);

    var z = try scaler.fit_transform(x);
    defer z.deinit(std.testing.allocator);
    var x_back = try scaler.inverse_transform(z);
    defer x_back.deinit(std.testing.allocator);

    for (0..x.rows) |i| {
        for (0..x.cols) |j| {
            try std.testing.expectApproxEqAbs(try x.get(i, j), try x_back.get(i, j), 1e-12);
        }
    }
}

test "MinMaxScaler default range" {
    const M = Matrix;
    var scaler = MinMaxScaler.init(std.testing.allocator);
    defer scaler.deinit();

    var x = try M.fromRowSlice(std.testing.allocator, 2, 2, &[_]f64{
        0.0, 10.0,
        2.0, 14.0,
    });
    defer x.deinit(std.testing.allocator);

    var z = try scaler.fit_transform(x);
    defer z.deinit(std.testing.allocator);
    try std.testing.expectApproxEqAbs(0.0, try z.get(0, 0), 1e-12);
    try std.testing.expectApproxEqAbs(0.0, try z.get(0, 1), 1e-12);
    try std.testing.expectApproxEqAbs(1.0, try z.get(1, 0), 1e-12);
    try std.testing.expectApproxEqAbs(1.0, try z.get(1, 1), 1e-12);
}

test "MinMaxScaler custom range" {
    const M = Matrix;
    var scaler = MinMaxScaler.init(std.testing.allocator);
    scaler.feature_min = -1.0;
    scaler.feature_max = 1.0;
    defer scaler.deinit();

    var x = try M.fromRowSlice(std.testing.allocator, 2, 1, &[_]f64{ 0.0, 2.0 });
    defer x.deinit(std.testing.allocator);

    var z = try scaler.fit_transform(x);
    defer z.deinit(std.testing.allocator);
    try std.testing.expectApproxEqAbs(-1.0, try z.get(0, 0), 1e-12);
    try std.testing.expectApproxEqAbs(1.0, try z.get(1, 0), 1e-12);
}

test "scalers reject unfitted transform" {
    var scaler = StandardScaler.init(std.testing.allocator);
    defer scaler.deinit();
    var x = try Matrix.init(std.testing.allocator, 2, 2);
    defer x.deinit(std.testing.allocator);
    try std.testing.expectError(error.NotFitted, scaler.transform(x));
}
