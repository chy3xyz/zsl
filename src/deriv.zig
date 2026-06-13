const std = @import("std");
const Error = @import("errors.zig").Error;

/// Function signature for univariate real-valued functions.
pub const Fn1D = *const fn (f64) f64;

/// Function signature for multivariate real-valued functions.
pub const FnND = *const fn ([]const f64) f64;

const epsilon = std.math.floatEps(f64);

fn central_deriv(f: Fn1D, x: f64, h: f64) struct { value: f64, trunc: f64, round: f64 } {
    // Compute the derivative using the 5-point rule (x-h, x-h/2, x,
    // x+h/2, x+h). Note that the central point is not used.
    // Compute the error using the difference between the 5-point and
    // the 3-point rule (x-h,x,x+h). Again the central point is not used.
    const fm1 = f(x - h);
    const fp1 = f(x + h);
    const fmh = f(x - h / 2.0);
    const fph = f(x + h / 2.0);
    const r3 = 0.50 * (fp1 - fm1);
    const r5 = (4.0 / 3.0) * (fph - fmh) - (1.0 / 3.0) * r3;
    const e3 = (@abs(fp1) + @abs(fm1)) * epsilon;
    const e5 = 2.0 * (@abs(fph) + @abs(fmh)) * epsilon + e3;
    const dy = @max(@abs(r3 / h), @abs(r5 / h)) * (@abs(x) / h) * epsilon;

    const result = r5 / h;
    const abserr_trunc = @abs((r5 - r3) / h); // Estimated truncation error O(h^2)
    const abserr_round = @abs(e5 / h) + dy; // Rounding error (cancellations)
    return .{ .value = result, .trunc = abserr_trunc, .round = abserr_round };
}

/// Compute the derivative of `f` at `x` using the central difference method.
pub fn central(f: Fn1D, x: f64, h: f64) struct { value: f64, err: f64 } {
    const d0 = central_deriv(f, x, h);
    var err = d0.round + d0.trunc;
    var result = d0.value;
    if (d0.round < d0.trunc and d0.round > 0.0 and d0.trunc > 0.0) {
        // Compute an optimised stepsize to minimize the total error,
        // using the scaling of the truncation error (O(h^2)) and
        // rounding error (O(1/h)).
        const h_opt = h * std.math.pow(f64, d0.round / (2.0 * d0.trunc), 1.0 / 3.0);
        const d_opt = central_deriv(f, x, h_opt);
        const err_opt = d_opt.round + d_opt.trunc;
        // Check that the new error is smaller, and that the new derivative
        // is consistent with the error bounds of the original estimate.
        if (err_opt < err and @abs(d_opt.value - d0.value) < 4.0 * err) {
            result = d_opt.value;
            err = err_opt;
        }
    }
    return .{ .value = result, .err = err };
}

fn forward_deriv(f: Fn1D, x: f64, h: f64) struct { value: f64, trunc: f64, round: f64 } {
    // Compute the derivative using the 4-point rule (x+h/4, x+h/2,
    // x+3h/4, x+h).
    // Compute the error using the difference between the 4-point and
    // the 2-point rule (x+h/2,x+h).
    const f1 = f(x + h / 4.0);
    const f2 = f(x + h / 2.0);
    const f3 = f(x + (3.0 / 4.0) * h);
    const f4 = f(x + h);
    const r2 = 2.0 * (f4 - f2);
    const r4 = (22.0 / 3.0) * (f4 - f3) - (62.0 / 3.0) * (f3 - f2) + (52.0 / 3.0) * (f2 - f1);
    const e4 = 2.0 * 20.670 * (@abs(f4) + @abs(f3) + @abs(f2) + @abs(f1)) * epsilon;
    const dy = @max(@abs(r2 / h), @abs(r4 / h)) * @abs(x / h) * epsilon;

    const result = r4 / h;
    const abserr_trunc = @abs((r4 - r2) / h); // Estimated truncation error O(h)
    const abserr_round = @abs(e4 / h) + dy;
    return .{ .value = result, .trunc = abserr_trunc, .round = abserr_round };
}

/// Compute the derivative of `f` at `x` using the forward difference method.
pub fn forward(f: Fn1D, x: f64, h: f64) struct { value: f64, err: f64 } {
    const d0 = forward_deriv(f, x, h);
    var err = d0.round + d0.trunc;
    var result = d0.value;
    if (d0.round < d0.trunc and d0.round > 0.0 and d0.trunc > 0.0) {
        // Compute an optimised stepsize to minimize the total error,
        // using the scaling of the estimated truncation error (O(h)) and
        // rounding error (O(1/h)).
        const h_opt = h * std.math.pow(f64, d0.round / d0.trunc, 1.0 / 2.0);
        const d_opt = forward_deriv(f, x, h_opt);
        const err_opt = d_opt.round + d_opt.trunc;
        // Check that the new error is smaller, and that the new derivative
        // is consistent with the error bounds of the original estimate.
        if (err_opt < err and @abs(d_opt.value - d0.value) < 4.0 * err) {
            result = d_opt.value;
            err = err_opt;
        }
    }
    return .{ .value = result, .err = err };
}

/// Compute the derivative of `f` at `x` using the backward difference method.
pub fn backward(f: Fn1D, x: f64, h: f64) struct { value: f64, err: f64 } {
    return forward(f, x, -h);
}

/// Compute the partial derivative of a multivariate function `f` at point
/// `x` with respect to the variable at index `variable`.
///
/// Note: Zig does not support closures, so instead of constructing a 1D
/// wrapper function that captures `f`, `x`, and `variable`, this function
/// applies the same 5-point central rule directly on the selected variable.
pub fn partial(f: FnND, x: []const f64, variable: usize, h: f64) Error!struct { value: f64, err: f64 } {
    if (variable >= x.len) {
        return error.IndexOutOfBounds;
    }

    const buffer = try std.heap.page_allocator.alloc(f64, x.len);
    defer std.heap.page_allocator.free(buffer);
    @memcpy(buffer, x);

    const xv = x[variable];

    buffer[variable] = xv - h;
    const fm1 = f(buffer);
    buffer[variable] = xv + h;
    const fp1 = f(buffer);
    buffer[variable] = xv - h / 2.0;
    const fmh = f(buffer);
    buffer[variable] = xv + h / 2.0;
    const fph = f(buffer);

    const r3 = 0.50 * (fp1 - fm1);
    const r5 = (4.0 / 3.0) * (fph - fmh) - (1.0 / 3.0) * r3;
    const e3 = (@abs(fp1) + @abs(fm1)) * epsilon;
    const e5 = 2.0 * (@abs(fph) + @abs(fmh)) * epsilon + e3;
    const dy = @max(@abs(r3 / h), @abs(r5 / h)) * (@abs(xv) / h) * epsilon;

    const result = r5 / h;
    const abserr_trunc = @abs((r5 - r3) / h);
    const abserr_round = @abs(e5 / h) + dy;

    var err = abserr_round + abserr_trunc;
    var value = result;
    if (abserr_round < abserr_trunc and abserr_round > 0.0 and abserr_trunc > 0.0) {
        const h_opt = h * std.math.pow(f64, abserr_round / (2.0 * abserr_trunc), 1.0 / 3.0);

        buffer[variable] = xv - h_opt;
        const fm1_opt = f(buffer);
        buffer[variable] = xv + h_opt;
        const fp1_opt = f(buffer);
        buffer[variable] = xv - h_opt / 2.0;
        const fmh_opt = f(buffer);
        buffer[variable] = xv + h_opt / 2.0;
        const fph_opt = f(buffer);

        const r3_opt = 0.50 * (fp1_opt - fm1_opt);
        const r5_opt = (4.0 / 3.0) * (fph_opt - fmh_opt) - (1.0 / 3.0) * r3_opt;
        const e3_opt = (@abs(fp1_opt) + @abs(fm1_opt)) * epsilon;
        const e5_opt = 2.0 * (@abs(fph_opt) + @abs(fmh_opt)) * epsilon + e3_opt;
        const dy_opt = @max(@abs(r3_opt / h_opt), @abs(r5_opt / h_opt)) * (@abs(xv) / h_opt) * epsilon;

        const result_opt = r5_opt / h_opt;
        const trunc_opt = @abs((r5_opt - r3_opt) / h_opt);
        const round_opt = @abs(e5_opt / h_opt) + dy_opt;
        const err_opt = round_opt + trunc_opt;

        if (err_opt < err and @abs(result_opt - result) < 4.0 * err) {
            value = result_opt;
            err = err_opt;
        }
    }

    return .{ .value = value, .err = err };
}

test "central difference for x^2 at x=3" {
    const f = struct {
        fn call(x: f64) f64 {
            return x * x;
        }
    }.call;
    const result = central(f, 3.0, 1.0e-4);
    try std.testing.expectApproxEqAbs(6.0, result.value, 1.0e-6);
}

test "central difference for sin(x) at x=0" {
    const f = struct {
        fn call(x: f64) f64 {
            return std.math.sin(x);
        }
    }.call;
    const result = central(f, 0.0, 1.0e-4);
    try std.testing.expectApproxEqAbs(1.0, result.value, 1.0e-6);
}

test "partial derivative of x*y at (2,3) w.r.t x" {
    const f = struct {
        fn call(v: []const f64) f64 {
            return v[0] * v[1];
        }
    }.call;
    const point = [_]f64{ 2.0, 3.0 };
    const result = try partial(f, &point, 0, 1.0e-4);
    try std.testing.expectApproxEqAbs(3.0, result.value, 1.0e-6);
}
