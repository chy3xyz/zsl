const std = @import("std");
const Error = @import("../errors.zig").Error;

fn checkLength(a: usize, b: usize) Error!void {
    if (a != b) return error.ShapeMismatch;
}

fn checkNotEmpty(len: usize) Error!void {
    if (len == 0) return error.InvalidDimension;
}

/// mean_squared_error computes the mean squared error (MSE).
pub fn mean_squared_error(y_true: []const f64, y_pred: []const f64) Error!f64 {
    try checkLength(y_true.len, y_pred.len);
    try checkNotEmpty(y_true.len);

    var sum: f64 = 0.0;
    for (y_true, y_pred) |yt, yp| {
        const diff = yt - yp;
        sum += diff * diff;
    }

    return sum / @as(f64, @floatFromInt(y_true.len));
}

/// root_mean_squared_error computes the root mean squared error (RMSE).
pub fn root_mean_squared_error(y_true: []const f64, y_pred: []const f64) Error!f64 {
    const mse = try mean_squared_error(y_true, y_pred);
    return std.math.sqrt(mse);
}

/// mean_absolute_error computes the mean absolute error (MAE).
pub fn mean_absolute_error(y_true: []const f64, y_pred: []const f64) Error!f64 {
    try checkLength(y_true.len, y_pred.len);
    try checkNotEmpty(y_true.len);

    var sum: f64 = 0.0;
    for (y_true, y_pred) |yt, yp| {
        sum += @abs(yt - yp);
    }

    return sum / @as(f64, @floatFromInt(y_true.len));
}

/// r2_score computes the coefficient of determination (R² score).
pub fn r2_score(y_true: []const f64, y_pred: []const f64) Error!f64 {
    try checkLength(y_true.len, y_pred.len);
    try checkNotEmpty(y_true.len);

    var sum: f64 = 0.0;
    for (y_true) |yt| sum += yt;
    const mean = sum / @as(f64, @floatFromInt(y_true.len));

    var ss_res: f64 = 0.0;
    var ss_tot: f64 = 0.0;
    for (y_true, y_pred) |yt, yp| {
        const diff_pred = yt - yp;
        const diff_mean = yt - mean;
        ss_res += diff_pred * diff_pred;
        ss_tot += diff_mean * diff_mean;
    }

    if (ss_tot == 0.0) {
        if (ss_res == 0.0) return 1.0;
        return 0.0;
    }

    return 1.0 - ss_res / ss_tot;
}

/// mean_absolute_percentage_error computes MAPE as a percentage.
pub fn mean_absolute_percentage_error(y_true: []const f64, y_pred: []const f64) Error!f64 {
    try checkLength(y_true.len, y_pred.len);
    try checkNotEmpty(y_true.len);

    var sum: f64 = 0.0;
    var valid_count: usize = 0;
    for (y_true, y_pred) |yt, yp| {
        if (yt != 0.0) {
            sum += @abs((yt - yp) / yt);
            valid_count += 1;
        }
    }

    if (valid_count == 0) return error.DivisionByZero;
    return (sum / @as(f64, @floatFromInt(valid_count))) * 100.0;
}

/// explained_variance_score computes the explained variance regression score.
pub fn explained_variance_score(y_true: []const f64, y_pred: []const f64) Error!f64 {
    try checkLength(y_true.len, y_pred.len);
    try checkNotEmpty(y_true.len);

    const n = @as(f64, @floatFromInt(y_true.len));

    var sum_res: f64 = 0.0;
    for (y_true, y_pred) |yt, yp| sum_res += yt - yp;
    const mean_res = sum_res / n;

    var var_res: f64 = 0.0;
    for (y_true, y_pred) |yt, yp| {
        const diff = (yt - yp) - mean_res;
        var_res += diff * diff;
    }
    var_res /= n;

    var sum_y: f64 = 0.0;
    for (y_true) |yt| sum_y += yt;
    const mean_y = sum_y / n;

    var var_y: f64 = 0.0;
    for (y_true) |yt| {
        const diff = yt - mean_y;
        var_y += diff * diff;
    }
    var_y /= n;

    if (var_y == 0.0) return 1.0;
    return 1.0 - var_res / var_y;
}

/// max_error computes the maximum absolute error.
pub fn max_error(y_true: []const f64, y_pred: []const f64) Error!f64 {
    try checkLength(y_true.len, y_pred.len);
    try checkNotEmpty(y_true.len);

    var max_err: f64 = 0.0;
    for (y_true, y_pred) |yt, yp| {
        const err = @abs(yt - yp);
        if (err > max_err) max_err = err;
    }

    return max_err;
}

/// median_absolute_error computes the median absolute error.
pub fn median_absolute_error(y_true: []const f64, y_pred: []const f64) Error!f64 {
    try checkLength(y_true.len, y_pred.len);
    try checkNotEmpty(y_true.len);

    var errors_ = try std.heap.smp_allocator.alloc(f64, y_true.len);
    defer std.heap.smp_allocator.free(errors_);
    for (y_true, y_pred, 0..) |yt, yp, i| {
        errors_[i] = @abs(yt - yp);
    }

    std.mem.sort(f64, errors_, {}, std.sort.asc(f64));

    const n = errors_.len;
    if (n % 2 == 0) {
        return (errors_[n / 2 - 1] + errors_[n / 2]) / 2.0;
    }
    return errors_[n / 2];
}

/// mean_squared_log_error computes the mean squared logarithmic error.
/// Only valid for non-negative values.
pub fn mean_squared_log_error(y_true: []const f64, y_pred: []const f64) Error!f64 {
    try checkLength(y_true.len, y_pred.len);
    try checkNotEmpty(y_true.len);

    var sum: f64 = 0.0;
    for (y_true, y_pred) |yt, yp| {
        if (yt < 0.0 or yp < 0.0) return error.InvalidDimension;
        const diff = @log(yt + 1.0) - @log(yp + 1.0);
        sum += diff * diff;
    }

    return sum / @as(f64, @floatFromInt(y_true.len));
}

test "mean_squared_error" {
    const y_true = &[_]f64{ 3.0, -0.5, 2.0, 7.0 };
    const y_pred = &[_]f64{ 2.5, 0.0, 2.0, 8.0 };
    try std.testing.expectApproxEqAbs(try mean_squared_error(y_true, y_pred), 0.375, 1e-12);
}

test "root_mean_squared_error" {
    const y_true = &[_]f64{ 3.0, -0.5, 2.0, 7.0 };
    const y_pred = &[_]f64{ 2.5, 0.0, 2.0, 8.0 };
    try std.testing.expectApproxEqAbs(try root_mean_squared_error(y_true, y_pred), std.math.sqrt(0.375), 1e-12);
}

test "mean_absolute_error" {
    const y_true = &[_]f64{ 3.0, -0.5, 2.0, 7.0 };
    const y_pred = &[_]f64{ 2.5, 0.0, 2.0, 8.0 };
    try std.testing.expectApproxEqAbs(try mean_absolute_error(y_true, y_pred), 0.5, 1e-12);
}

test "r2_score" {
    const y_true = &[_]f64{ 3.0, -0.5, 2.0, 7.0 };
    const y_pred = &[_]f64{ 2.5, 0.0, 2.0, 8.0 };
    try std.testing.expectApproxEqAbs(try r2_score(y_true, y_pred), 0.948608137, 1e-9);
}

test "mean_absolute_percentage_error" {
    const y_true = &[_]f64{ 100.0, 200.0, 300.0, 400.0 };
    const y_pred = &[_]f64{ 90.0, 210.0, 300.0, 430.0 };
    const mape = try mean_absolute_percentage_error(y_true, y_pred);
    try std.testing.expectApproxEqAbs(mape, 5.625, 1e-12);
}

test "explained_variance_score" {
    const y_true = &[_]f64{ 3.0, -0.5, 2.0, 7.0 };
    const y_pred = &[_]f64{ 2.5, 0.0, 2.0, 8.0 };
    try std.testing.expectApproxEqAbs(try explained_variance_score(y_true, y_pred), 0.9571734475, 1e-9);
}

test "max_error" {
    const y_true = &[_]f64{ 3.0, -0.5, 2.0, 7.0 };
    const y_pred = &[_]f64{ 2.5, 0.0, 2.0, 8.0 };
    try std.testing.expectApproxEqAbs(try max_error(y_true, y_pred), 1.0, 1e-12);
}

test "median_absolute_error" {
    const y_true = &[_]f64{ 3.0, -0.5, 2.0, 7.0 };
    const y_pred = &[_]f64{ 2.5, 0.0, 2.0, 8.0 };
    try std.testing.expectApproxEqAbs(try median_absolute_error(y_true, y_pred), 0.5, 1e-12);
}

test "mean_squared_log_error" {
    const y_true = &[_]f64{ 3.0, 5.0, 2.5, 7.0 };
    const y_pred = &[_]f64{ 2.5, 5.0, 4.0, 8.0 };
    const msle = try mean_squared_log_error(y_true, y_pred);
    const expected = ((@log(3.0 + 1.0) - @log(2.5 + 1.0)) *
        (@log(3.0 + 1.0) - @log(2.5 + 1.0)) +
        (@log(2.5 + 1.0) - @log(4.0 + 1.0)) *
            (@log(2.5 + 1.0) - @log(4.0 + 1.0)) +
        (@log(7.0 + 1.0) - @log(8.0 + 1.0)) *
            (@log(7.0 + 1.0) - @log(8.0 + 1.0))) / 4.0;
    try std.testing.expectApproxEqAbs(msle, expected, 1e-12);
}
