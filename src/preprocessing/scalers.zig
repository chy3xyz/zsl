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

/// RobustScaler scales features using statistics that are robust to outliers.
/// For each feature it subtracts the median and divides by the interquartile
/// range (IQR) defined by `quantile_range` (default [0.25, 0.75]).
pub const RobustScaler = struct {
    allocator: std.mem.Allocator,
    fitted: bool,
    median_: []f64,
    iqr_: []f64,
    n_features: usize,
    quantile_range: [2]f64,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .fitted = false,
            .median_ = &[_]f64{},
            .iqr_ = &[_]f64{},
            .n_features = 0,
            .quantile_range = .{ 0.25, 0.75 },
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.median_);
        self.allocator.free(self.iqr_);
        self.median_ = &[_]f64{};
        self.iqr_ = &[_]f64{};
        self.fitted = false;
        self.n_features = 0;
    }

    fn validateQuantileRange(self: Self) Error!void {
        if (self.quantile_range[0] < 0.0 or self.quantile_range[1] > 1.0 or
            self.quantile_range[0] >= self.quantile_range[1])
        {
            return error.InvalidDimension;
        }
    }

    fn quantile(sorted: []const f64, p: f64) f64 {
        const idx = @as(usize, @intFromFloat(p * @as(f64, @floatFromInt(sorted.len - 1))));
        return sorted[idx];
    }

    pub fn fit(self: *Self, x: Matrix) Error!void {
        if (x.rows == 0 or x.cols == 0) return error.InvalidDimension;
        try self.validateQuantileRange();

        self.deinit();
        self.n_features = x.cols;
        self.median_ = try self.allocator.alloc(f64, self.n_features);
        errdefer self.allocator.free(self.median_);
        self.iqr_ = try self.allocator.alloc(f64, self.n_features);
        errdefer self.allocator.free(self.iqr_);
        @memset(self.median_, 0);
        @memset(self.iqr_, 1);

        var col = try self.allocator.alloc(f64, x.rows);
        defer self.allocator.free(col);

        for (0..self.n_features) |j| {
            for (0..x.rows) |i| {
                col[i] = try x.get(i, j);
            }
            std.mem.sort(f64, col, {}, std.sort.asc(f64));

            // Median.
            if (x.rows % 2 == 0) {
                self.median_[j] = (col[x.rows / 2 - 1] + col[x.rows / 2]) / 2.0;
            } else {
                self.median_[j] = col[x.rows / 2];
            }

            // IQR from configured quantiles.
            const q1 = quantile(col, self.quantile_range[0]);
            const q3 = quantile(col, self.quantile_range[1]);
            const iqr = q3 - q1;

            // Leave the feature unchanged when the IQR is zero by setting
            // median to 0 and scale to 1, so (x - 0) / 1 == x.
            if (iqr == 0.0) {
                self.median_[j] = 0.0;
                self.iqr_[j] = 1.0;
            } else {
                self.iqr_[j] = iqr;
            }
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
                try result.set(i, j, (v - self.median_[j]) / self.iqr_[j]);
            }
        }
        return result;
    }

    pub fn fit_transform(self: *Self, x: Matrix) Error!Matrix {
        try self.fit(x);
        return try self.transform(x);
    }
};

test "scalers reject unfitted transform" {
    var scaler = StandardScaler.init(std.testing.allocator);
    defer scaler.deinit();
    var x = try Matrix.init(std.testing.allocator, 2, 2);
    defer x.deinit(std.testing.allocator);
    try std.testing.expectError(error.NotFitted, scaler.transform(x));
}

test "RobustScaler default quantile range" {
    const M = Matrix;
    var scaler = RobustScaler.init(std.testing.allocator);
    defer scaler.deinit();

    var x = try M.fromRowSlice(std.testing.allocator, 5, 1, &[_]f64{
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    });
    defer x.deinit(std.testing.allocator);

    try scaler.fit(x);
    try std.testing.expect(scaler.fitted);
    try std.testing.expectApproxEqAbs(3.0, scaler.median_[0], 1e-12);
    // Q1 = col[1] = 2, Q3 = col[3] = 4, IQR = 2.
    try std.testing.expectApproxEqAbs(2.0, scaler.iqr_[0], 1e-12);

    var z = try scaler.transform(x);
    defer z.deinit(std.testing.allocator);
    try std.testing.expectApproxEqAbs(-1.0, try z.get(0, 0), 1e-12);
    try std.testing.expectApproxEqAbs(0.0, try z.get(2, 0), 1e-12);
    try std.testing.expectApproxEqAbs(1.0, try z.get(4, 0), 1e-12);
}

test "RobustScaler outlier robustness" {
    const M = Matrix;
    var scaler = RobustScaler.init(std.testing.allocator);
    defer scaler.deinit();

    var x = try M.fromRowSlice(std.testing.allocator, 6, 1, &[_]f64{
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        100.0,
    });
    defer x.deinit(std.testing.allocator);

    var z = try scaler.fit_transform(x);
    defer z.deinit(std.testing.allocator);
    // Median of [1..5,100] = (3+4)/2 = 3.5; IQR = Q3-Q1 = col[3]-col[1] = 4-2 = 2.
    try std.testing.expectApproxEqAbs(3.5, scaler.median_[0], 1e-12);
    try std.testing.expectApproxEqAbs(2.0, scaler.iqr_[0], 1e-12);
    try std.testing.expectApproxEqAbs(-1.25, try z.get(0, 0), 1e-12);
    try std.testing.expectApproxEqAbs(48.25, try z.get(5, 0), 1e-12);
}

test "RobustScaler zero IQR leaves feature unchanged" {
    const M = Matrix;
    var scaler = RobustScaler.init(std.testing.allocator);
    defer scaler.deinit();

    var x = try M.fromRowSlice(std.testing.allocator, 3, 2, &[_]f64{
        5.0, 1.0,
        5.0, 2.0,
        5.0, 3.0,
    });
    defer x.deinit(std.testing.allocator);

    var z = try scaler.fit_transform(x);
    defer z.deinit(std.testing.allocator);
    // First feature is constant, so it must pass through unchanged.
    try std.testing.expectApproxEqAbs(5.0, try z.get(0, 0), 1e-12);
    try std.testing.expectApproxEqAbs(5.0, try z.get(1, 0), 1e-12);
    try std.testing.expectApproxEqAbs(5.0, try z.get(2, 0), 1e-12);
    // Second feature is scaled normally.
    try std.testing.expectApproxEqAbs(-1.0, try z.get(0, 1), 1e-12);
    try std.testing.expectApproxEqAbs(0.0, try z.get(1, 1), 1e-12);
    try std.testing.expectApproxEqAbs(1.0, try z.get(2, 1), 1e-12);
}

test "RobustScaler custom quantile range" {
    const M = Matrix;
    var scaler = RobustScaler.init(std.testing.allocator);
    scaler.quantile_range = .{ 0.1, 0.9 };
    defer scaler.deinit();

    var x = try M.fromRowSlice(std.testing.allocator, 5, 1, &[_]f64{
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    });
    defer x.deinit(std.testing.allocator);

    try scaler.fit(x);
    // 0.1 quantile index = int(0.1*4) = 0 -> 1.0
    // 0.9 quantile index = int(0.9*4) = 3 -> 4.0
    try std.testing.expectApproxEqAbs(3.0, scaler.iqr_[0], 1e-12);
}

test "RobustScaler rejects invalid quantile range" {
    const M = Matrix;
    var scaler = RobustScaler.init(std.testing.allocator);
    scaler.quantile_range = .{ 0.75, 0.25 };
    defer scaler.deinit();

    var x = try M.fromRowSlice(std.testing.allocator, 3, 1, &[_]f64{ 1.0, 2.0, 3.0 });
    defer x.deinit(std.testing.allocator);

    try std.testing.expectError(error.InvalidDimension, scaler.fit(x));
}
