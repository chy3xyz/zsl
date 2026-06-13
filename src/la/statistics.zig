const std = @import("std");
const la = @import("../la.zig");
const util = @import("../util.zig");
const Error = @import("../errors.zig").Error;
const Matrix = la.Matrix;

/// Pearson correlation matrix of a data matrix.
/// Rows are samples and columns are features; result is n x n.
pub fn correlation_matrix(data: Matrix(f64), allocator: std.mem.Allocator) Error!Matrix(f64) {
    _ = util.Float(f64);
    const m = data.rows;
    const n = data.cols;
    if (m == 0 or n == 0) return error.InvalidDimension;

    const means = try column_mean(data, allocator);
    defer allocator.free(means);

    // Population standard deviations with zero values replaced by 1.0.
    const stds = try allocator.alloc(f64, n);
    defer allocator.free(stds);

    for (0..n) |j| {
        var sum_sq: f64 = 0.0;
        for (0..m) |i| {
            const diff = try data.get(i, j) - means[j];
            sum_sq += diff * diff;
        }
        stds[j] = std.math.sqrt(sum_sq / @as(f64, @floatFromInt(m)));
        if (stds[j] == 0.0) {
            stds[j] = 1.0;
        }
    }

    var corr = try Matrix(f64).init(allocator, n, n);
    errdefer corr.deinit(allocator);

    for (0..n) |i| {
        for (0..n) |j| {
            if (i == j) {
                try corr.set(i, j, 1.0);
            } else if (j > i) {
                var sum: f64 = 0.0;
                for (0..m) |k| {
                    const xi = (try data.get(k, i) - means[i]) / stds[i];
                    const xj = (try data.get(k, j) - means[j]) / stds[j];
                    sum += xi * xj;
                }
                const r = sum / @as(f64, @floatFromInt(m));
                try corr.set(i, j, r);
                try corr.set(j, i, r);
            }
        }
    }

    return corr;
}

/// Covariance matrix of a data matrix.
/// ddof = 0 gives the population covariance; ddof = 1 gives the sample covariance.
pub fn covariance_matrix(data: Matrix(f64), ddof: usize, allocator: std.mem.Allocator) Error!Matrix(f64) {
    _ = util.Float(f64);
    const m = data.rows;
    const n = data.cols;
    if (m == 0 or n == 0) return error.InvalidDimension;
    if (ddof > m) return error.InvalidDimension;

    const means = try column_mean(data, allocator);
    defer allocator.free(means);

    var cov = try Matrix(f64).init(allocator, n, n);
    errdefer cov.deinit(allocator);

    const divisor = @as(f64, @floatFromInt(m - ddof));
    if (divisor == 0.0) return error.DivisionByZero;

    for (0..n) |i| {
        for (i..n) |j| {
            var sum: f64 = 0.0;
            for (0..m) |k| {
                const di = try data.get(k, i) - means[i];
                const dj = try data.get(k, j) - means[j];
                sum += di * dj;
            }
            const c = sum / divisor;
            try cov.set(i, j, c);
            if (i != j) {
                try cov.set(j, i, c);
            }
        }
    }

    return cov;
}

/// Mean of each column.
pub fn column_mean(data: Matrix(f64), allocator: std.mem.Allocator) Error![]f64 {
    _ = util.Float(f64);
    const m = data.rows;
    const n = data.cols;
    if (m == 0 or n == 0) return error.InvalidDimension;

    const means = try allocator.alloc(f64, n);
    errdefer allocator.free(means);

    for (0..n) |j| {
        var sum: f64 = 0.0;
        for (0..m) |i| {
            sum += try data.get(i, j);
        }
        means[j] = sum / @as(f64, @floatFromInt(m));
    }

    return means;
}

/// Standard deviation of each column.
/// ddof = 0 gives the population standard deviation; ddof = 1 gives the sample standard deviation.
pub fn column_std(data: Matrix(f64), ddof: usize, allocator: std.mem.Allocator) Error![]f64 {
    _ = util.Float(f64);
    const m = data.rows;
    const n = data.cols;
    if (m == 0 or n == 0) return error.InvalidDimension;
    if (ddof > m) return error.InvalidDimension;

    const means = try column_mean(data, allocator);
    defer allocator.free(means);

    const stds = try allocator.alloc(f64, n);
    errdefer allocator.free(stds);

    const divisor = @as(f64, @floatFromInt(m - ddof));
    if (divisor == 0.0) return error.DivisionByZero;

    for (0..n) |j| {
        var sum_sq: f64 = 0.0;
        for (0..m) |i| {
            const diff = try data.get(i, j) - means[j];
            sum_sq += diff * diff;
        }
        stds[j] = std.math.sqrt(sum_sq / divisor);
    }

    return stds;
}

/// Minimum value of each column.
pub fn column_min(data: Matrix(f64), allocator: std.mem.Allocator) Error![]f64 {
    _ = util.Float(f64);
    const m = data.rows;
    const n = data.cols;
    if (m == 0 or n == 0) return error.InvalidDimension;

    const mins = try allocator.alloc(f64, n);
    errdefer allocator.free(mins);

    @memset(mins, std.math.inf(f64));

    for (0..n) |j| {
        for (0..m) |i| {
            const val = try data.get(i, j);
            if (val < mins[j]) {
                mins[j] = val;
            }
        }
    }

    return mins;
}

/// Maximum value of each column.
pub fn column_max(data: Matrix(f64), allocator: std.mem.Allocator) Error![]f64 {
    _ = util.Float(f64);
    const m = data.rows;
    const n = data.cols;
    if (m == 0 or n == 0) return error.InvalidDimension;

    const maxs = try allocator.alloc(f64, n);
    errdefer allocator.free(maxs);

    @memset(maxs, -std.math.inf(f64));

    for (0..n) |j| {
        for (0..m) |i| {
            const val = try data.get(i, j);
            if (val > maxs[j]) {
                maxs[j] = val;
            }
        }
    }

    return maxs;
}

/// Sum of each column.
pub fn column_sum(data: Matrix(f64), allocator: std.mem.Allocator) Error![]f64 {
    _ = util.Float(f64);
    const m = data.rows;
    const n = data.cols;
    if (m == 0 or n == 0) return error.InvalidDimension;

    const sums = try allocator.alloc(f64, n);
    errdefer allocator.free(sums);

    @memset(sums, 0.0);

    for (0..n) |j| {
        for (0..m) |i| {
            sums[j] += try data.get(i, j);
        }
    }

    return sums;
}

/// Center each column by subtracting its mean.
pub fn center_matrix(data: Matrix(f64), allocator: std.mem.Allocator) Error!Matrix(f64) {
    _ = util.Float(f64);
    const m = data.rows;
    const n = data.cols;
    if (m == 0 or n == 0) return error.InvalidDimension;

    const means = try column_mean(data, allocator);
    defer allocator.free(means);

    var centered = try Matrix(f64).init(allocator, m, n);
    errdefer centered.deinit(allocator);

    for (0..m) |i| {
        for (0..n) |j| {
            try centered.set(i, j, try data.get(i, j) - means[j]);
        }
    }

    return centered;
}

/// Standardize each column using the z-score (population std).
pub fn standardize_matrix(data: Matrix(f64), allocator: std.mem.Allocator) Error!Matrix(f64) {
    _ = util.Float(f64);
    const m = data.rows;
    const n = data.cols;
    if (m == 0 or n == 0) return error.InvalidDimension;

    const means = try column_mean(data, allocator);
    defer allocator.free(means);

    const stds = try column_std(data, 0, allocator);
    defer allocator.free(stds);

    var standardized = try Matrix(f64).init(allocator, m, n);
    errdefer standardized.deinit(allocator);

    for (0..m) |i| {
        for (0..n) |j| {
            const std_val = if (stds[j] == 0.0) 1.0 else stds[j];
            try standardized.set(i, j, (try data.get(i, j) - means[j]) / std_val);
        }
    }

    return standardized;
}

test "column_mean, column_std, column_sum, min and max" {
    const float = @import("../float.zig");
    const M = Matrix(f64);

    var data = try M.fromRowSlice(std.testing.allocator, 4, 2, &[_]f64{
        1.0, 2.0,
        2.0, 1.0,
        3.0, 4.0,
        4.0, 3.0,
    });
    defer data.deinit(std.testing.allocator);

    const means = try column_mean(data, std.testing.allocator);
    defer std.testing.allocator.free(means);
    try std.testing.expect(float.approxEqAbs(f64, means[0], 2.5, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, means[1], 2.5, 1e-12));

    const stds_pop = try column_std(data, 0, std.testing.allocator);
    defer std.testing.allocator.free(stds_pop);
    try std.testing.expect(float.approxEqAbs(f64, stds_pop[0], std.math.sqrt(1.25), 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, stds_pop[1], std.math.sqrt(1.25), 1e-12));

    const stds_sample = try column_std(data, 1, std.testing.allocator);
    defer std.testing.allocator.free(stds_sample);
    try std.testing.expect(float.approxEqAbs(f64, stds_sample[0], std.math.sqrt(5.0 / 3.0), 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, stds_sample[1], std.math.sqrt(5.0 / 3.0), 1e-12));

    const sums = try column_sum(data, std.testing.allocator);
    defer std.testing.allocator.free(sums);
    try std.testing.expect(float.approxEqAbs(f64, sums[0], 10.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, sums[1], 10.0, 1e-12));

    const mins = try column_min(data, std.testing.allocator);
    defer std.testing.allocator.free(mins);
    try std.testing.expect(float.approxEqAbs(f64, mins[0], 1.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, mins[1], 1.0, 1e-12));

    const maxs = try column_max(data, std.testing.allocator);
    defer std.testing.allocator.free(maxs);
    try std.testing.expect(float.approxEqAbs(f64, maxs[0], 4.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, maxs[1], 4.0, 1e-12));
}

test "covariance_matrix population and sample" {
    const float = @import("../float.zig");
    const M = Matrix(f64);

    var data = try M.fromRowSlice(std.testing.allocator, 4, 2, &[_]f64{
        1.0, 2.0,
        2.0, 1.0,
        3.0, 4.0,
        4.0, 3.0,
    });
    defer data.deinit(std.testing.allocator);

    var cov_pop = try covariance_matrix(data, 0, std.testing.allocator);
    defer cov_pop.deinit(std.testing.allocator);
    try std.testing.expect(float.approxEqAbs(f64, try cov_pop.get(0, 0), 1.25, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try cov_pop.get(1, 1), 1.25, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try cov_pop.get(0, 1), 0.75, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try cov_pop.get(1, 0), 0.75, 1e-12));

    var cov_sample = try covariance_matrix(data, 1, std.testing.allocator);
    defer cov_sample.deinit(std.testing.allocator);
    try std.testing.expect(float.approxEqAbs(f64, try cov_sample.get(0, 0), 5.0 / 3.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try cov_sample.get(1, 1), 5.0 / 3.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try cov_sample.get(0, 1), 1.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try cov_sample.get(1, 0), 1.0, 1e-12));
}

test "correlation_matrix" {
    const float = @import("../float.zig");
    const M = Matrix(f64);

    var data = try M.fromRowSlice(std.testing.allocator, 4, 2, &[_]f64{
        1.0, 2.0,
        2.0, 1.0,
        3.0, 4.0,
        4.0, 3.0,
    });
    defer data.deinit(std.testing.allocator);

    var corr = try correlation_matrix(data, std.testing.allocator);
    defer corr.deinit(std.testing.allocator);
    try std.testing.expect(float.approxEqAbs(f64, try corr.get(0, 0), 1.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try corr.get(1, 1), 1.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try corr.get(0, 1), 0.6, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try corr.get(1, 0), 0.6, 1e-12));
}

test "center_matrix and standardize_matrix" {
    const float = @import("../float.zig");
    const M = Matrix(f64);

    var data = try M.fromRowSlice(std.testing.allocator, 4, 2, &[_]f64{
        1.0, 2.0,
        2.0, 1.0,
        3.0, 4.0,
        4.0, 3.0,
    });
    defer data.deinit(std.testing.allocator);

    var centered = try center_matrix(data, std.testing.allocator);
    defer centered.deinit(std.testing.allocator);
    try std.testing.expect(float.approxEqAbs(f64, try centered.get(0, 0), -1.5, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try centered.get(0, 1), -0.5, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try centered.get(1, 0), -0.5, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try centered.get(1, 1), -1.5, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try centered.get(2, 0), 0.5, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try centered.get(2, 1), 1.5, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try centered.get(3, 0), 1.5, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try centered.get(3, 1), 0.5, 1e-12));

    var standardized = try standardize_matrix(data, std.testing.allocator);
    defer standardized.deinit(std.testing.allocator);
    const s = std.math.sqrt(1.25);
    try std.testing.expect(float.approxEqAbs(f64, try standardized.get(0, 0), -1.5 / s, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try standardized.get(0, 1), -0.5 / s, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try standardized.get(1, 0), -0.5 / s, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try standardized.get(1, 1), -1.5 / s, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try standardized.get(2, 0), 0.5 / s, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try standardized.get(2, 1), 1.5 / s, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try standardized.get(3, 0), 1.5 / s, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try standardized.get(3, 1), 0.5 / s, 1e-12));
}
