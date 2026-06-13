const std = @import("std");

/// Function signature for one-dimensional real-valued functions.
pub const Fn1D = *const fn (f64) f64;

const sqrt_eps = std.math.sqrt(std.math.floatEps(f64));

/// Numerical derivative of `f` at `x` using the backward difference formula.
/// Returns the estimated derivative and an error estimate.
pub fn backward(f: Fn1D, x: f64) struct { value: f64, err: f64 } {
    var h = sqrt_eps;

    // Construct a divided difference table with a fairly large step size to
    // get a very rough estimate of f''. Use this to estimate the step size
    // which will minimize the error in calculating f'.
    //
    // Algorithm based on description on pg. 204 of Conte and de Boor (CdB) -
    // coefficients of Newton form of polynomial of degree 2.
    var a: [3]f64 = undefined;
    var d: [3]f64 = undefined;
    for (0..3) |ii| {
        const i: f64 = @floatFromInt(ii);
        a[ii] = x + (i - 2.0) * h;
        d[ii] = f(a[ii]);
    }
    for (1..4) |kk| {
        for (0..3 - kk) |ii| {
            d[ii] = (d[ii + 1] - d[ii]) / (a[ii + kk] - a[ii]);
        }
    }

    // Adapt procedure described on pg. 282 of CdB to find best value of step
    // size.
    var a2 = @abs(d[0] + d[1] + d[2]);
    if (a2 < 100.0 * sqrt_eps) {
        a2 = 100.0 * sqrt_eps;
    }
    h = std.math.sqrt(sqrt_eps / (2.0 * a2));
    if (h > 100.0 * sqrt_eps) {
        h = 100.0 * sqrt_eps;
    }

    return .{
        .value = (f(x) - f(x - h)) / h,
        .err = @abs(10.0 * a2 * h),
    };
}

/// Numerical derivative of `f` at `x` using the forward difference formula.
/// Returns the estimated derivative and an error estimate.
pub fn forward(f: Fn1D, x: f64) struct { value: f64, err: f64 } {
    var h = sqrt_eps;

    // Construct a divided difference table with a fairly large step size to
    // get a very rough estimate of f''. Use this to estimate the step size
    // which will minimize the error in calculating f'.
    var a: [3]f64 = undefined;
    var d: [3]f64 = undefined;
    for (0..3) |ii| {
        const i: f64 = @floatFromInt(ii);
        a[ii] = x + i * h;
        d[ii] = f(a[ii]);
    }
    for (1..4) |kk| {
        for (0..3 - kk) |ii| {
            d[ii] = (d[ii + 1] - d[ii]) / (a[ii + kk] - a[ii]);
        }
    }

    // Adapt procedure described on pg. 282 of CdB to find best value of step
    // size.
    var a2 = @abs(d[0] + d[1] + d[2]);
    if (a2 < 100.0 * sqrt_eps) {
        a2 = 100.0 * sqrt_eps;
    }
    h = std.math.sqrt(sqrt_eps / (2.0 * a2));
    if (h > 100.0 * sqrt_eps) {
        h = 100.0 * sqrt_eps;
    }

    return .{
        .value = (f(x + h) - f(x)) / h,
        .err = @abs(10.0 * a2 * h),
    };
}

/// Numerical derivative of `f` at `x` using the central difference formula.
/// Returns the estimated derivative and an error estimate.
pub fn central(f: Fn1D, x: f64) struct { value: f64, err: f64 } {
    var h = sqrt_eps;

    // Construct a divided difference table with a fairly large step size to
    // get a very rough estimate of f'''. Use this to estimate the step size
    // which will minimize the error in calculating f'.
    //
    // Algorithm based on description on pg. 204 of Conte and de Boor (CdB) -
    // coefficients of Newton form of polynomial of degree 3.
    var a: [4]f64 = undefined;
    var d: [4]f64 = undefined;
    for (0..4) |ii| {
        const i: f64 = @floatFromInt(ii);
        a[ii] = x + (i - 2.0) * h;
        d[ii] = f(a[ii]);
    }
    for (1..5) |kk| {
        for (0..4 - kk) |ii| {
            d[ii] = (d[ii + 1] - d[ii]) / (a[ii + kk] - a[ii]);
        }
    }

    // Adapt procedure described on pg. 282 of CdB to find best value of step
    // size.
    var a3 = @abs(d[0] + d[1] + d[2] + d[3]);
    if (a3 < 100.0 * sqrt_eps) {
        a3 = 100.0 * sqrt_eps;
    }
    h = std.math.pow(f64, sqrt_eps / (2.0 * a3), 1.0 / 3.0);
    if (h > 100.0 * sqrt_eps) {
        h = 100.0 * sqrt_eps;
    }

    return .{
        .value = (f(x + h) - f(x - h)) / (2.0 * h),
        .err = @abs(100.0 * a3 * h * h),
    };
}

test "backward differentiates x^2 at x=3" {
    const f = struct {
        fn call(x: f64) f64 {
            return x * x;
        }
    }.call;
    const result = backward(f, 3.0);
    try std.testing.expectApproxEqAbs(result.value, 6.0, 1e-5);
}

test "forward differentiates x^2 at x=3" {
    const f = struct {
        fn call(x: f64) f64 {
            return x * x;
        }
    }.call;
    const result = forward(f, 3.0);
    try std.testing.expectApproxEqAbs(result.value, 6.0, 1e-5);
}

test "central differentiates x^2 at x=3" {
    const f = struct {
        fn call(x: f64) f64 {
            return x * x;
        }
    }.call;
    const result = central(f, 3.0);
    try std.testing.expectApproxEqAbs(result.value, 6.0, 1e-6);
}

test "central differentiates sin(x) at x=0" {
    const f = struct {
        fn call(x: f64) f64 {
            return std.math.sin(x);
        }
    }.call;
    const result = central(f, 0.0);
    try std.testing.expectApproxEqAbs(result.value, 1.0, 1e-6);
}
