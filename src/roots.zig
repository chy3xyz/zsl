const std = @import("std");
const Error = @import("errors.zig").Error;

/// Find a root of `f` within the bracket `[a, b]` using the bisection method.
/// The interval must bracket a root (i.e. `f(a)` and `f(b)` have opposite signs).
/// Returns `error.NotConverged` if no root is found within `max_iter`.
pub fn bisection(f: *const fn (f64) f64, a: f64, b: f64, tol: f64, max_iter: usize) Error!f64 {
    var lo = a;
    var hi = b;
    var flo = f(lo);
    var fhi = f(hi);

    if (flo == 0.0) return lo;
    if (fhi == 0.0) return hi;
    if (flo * fhi > 0.0) return error.NotConverged;

    var i: usize = 0;
    while (i < max_iter) : (i += 1) {
        const mid = 0.5 * (lo + hi);
        const fmid = f(mid);
        if (@abs(fmid) < tol or @abs(hi - lo) < tol) {
            return mid;
        }
        if (fmid == 0.0) return mid;
        if (flo * fmid < 0.0) {
            hi = mid;
            fhi = fmid;
        } else {
            lo = mid;
            flo = fmid;
        }
    }
    return error.NotConverged;
}

/// Find a root of `f` within the bracket `[a, b]` using Brent's method.
/// The interval must bracket a root. Returns `error.NotConverged` if the
/// method fails to converge within `max_iter`.
pub fn brent(f: *const fn (f64) f64, a: f64, b: f64, tol: f64, max_iter: usize) Error!f64 {
    var aa = a;
    var bb = b;
    var c = aa;
    var fa = f(aa);
    var fb = f(bb);
    var fc = fa;

    if (fa == 0.0) return aa;
    if (fb == 0.0) return bb;
    if ((fa > 0.0 and fb > 0.0) or (fa < 0.0 and fb < 0.0)) {
        return error.NotConverged;
    }

    var prev_step = bb - aa;
    var tol1: f64 = tol;
    var p: f64 = 0.0;
    var q: f64 = 0.0;
    var r: f64 = 0.0;

    var iter: usize = 0;
    while (iter < max_iter) : (iter += 1) {
        prev_step = bb - aa;
        if (@abs(fc) < @abs(fb)) {
            aa = bb;
            bb = c;
            c = aa;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        tol1 = 2.0 * std.math.floatEps(f64) * @abs(bb) + 0.5 * tol;
        var new_step = 0.5 * (c - bb);
        if (@abs(new_step) <= tol1 or fb == 0.0) {
            return bb;
        }
        if (@abs(prev_step) >= tol1 and @abs(fa) > @abs(fb)) {
            const s = fb / fa;
            if (aa == c) {
                p = 2.0 * new_step * s;
                q = 1.0 - s;
            } else {
                q = fa / fc;
                r = fb / fc;
                p = s * (2.0 * new_step * q * (q - r) - (bb - aa) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) {
                q = -q;
            } else {
                p = -p;
            }
            if (2.0 * p < 3.0 * new_step * q - @abs(tol1 * q) and
                2.0 * p < @abs(prev_step * q))
            {
                new_step = p / q;
            } else {
                new_step = 0.5 * (c - bb);
                prev_step = new_step;
            }
        }
        if (@abs(new_step) < tol1) {
            new_step = if (new_step > 0.0) tol1 else -tol1;
        }
        aa = bb;
        fa = fb;
        bb += new_step;
        fb = f(bb);
        if ((fb < 0.0 and fc < 0.0) or (fb > 0.0 and fc > 0.0)) {
            c = aa;
            fc = fa;
        }
    }
    return error.NotConverged;
}

/// Find a root of `f` starting from `x0` using Newton's method with an
/// Armijo line search. `df` is the derivative of `f`.
/// Returns `error.NotConverged` if the method fails within `max_iter`.
pub fn newton(f: *const fn (f64) f64, df: *const fn (f64) f64, x0: f64, tol: f64, max_iter: usize) Error!f64 {
    const omega = 1e-4;
    const gamma = 0.5;
    var root = x0;
    var fval = f(root);
    var dfval = df(root);

    var i: usize = 0;
    while (i < max_iter) : (i += 1) {
        if (dfval == 0.0) return error.NotConverged;
        const dx = fval / dfval;
        const norm0 = @abs(fval);
        var t: f64 = 1.0;
        var norm: f64 = 0.0;
        while (t != 0.0) {
            const x_linesearch = root - t * dx;
            fval = f(x_linesearch);
            dfval = df(x_linesearch);
            norm = @abs(fval);
            if (norm < norm0 * (1.0 - omega * t)) {
                root = x_linesearch;
                break;
            }
            t *= gamma;
        }
        if (@abs(dx) < tol * @abs(root) or norm < tol) {
            break;
        }
    }
    if (i == max_iter) return error.NotConverged;
    return root;
}

/// Find a root of `f` within the bracket `[a, b]` by combining Newton's
/// method with bisection. `df` is the derivative of `f`.
/// Returns `error.NotConverged` if the method fails within `max_iter`.
pub fn newton_bisection(f: *const fn (f64) f64, df: *const fn (f64) f64, a: f64, b: f64, tol: f64, max_iter: usize) Error!f64 {
    const func_low = f(a);
    if (func_low == 0.0) return a;
    const func_high = f(b);
    if (func_high == 0.0) return b;
    if ((func_low > 0.0 and func_high > 0.0) or (func_low < 0.0 and func_high < 0.0)) {
        return error.NotConverged;
    }

    var xl: f64 = undefined;
    var xh: f64 = undefined;
    if (func_low < 0.0) {
        xl = a;
        xh = b;
    } else {
        xl = b;
        xh = a;
    }

    var rts = 0.5 * (a + b);
    var dx_anc = @abs(b - a);
    var dx = dx_anc;
    var func_current = f(rts);
    var diff_func_current = df(rts);

    var i: usize = 0;
    while (i < max_iter) : (i += 1) {
        if (((rts - xh) * diff_func_current - func_current) *
            ((rts - xl) * diff_func_current - func_current) >= 0.0 or
            @abs(2.0 * func_current) > @abs(dx_anc * diff_func_current))
        {
            dx_anc = dx;
            dx = 0.5 * (xh - xl);
            rts = xl + dx;
        } else {
            dx_anc = dx;
            dx = func_current / diff_func_current;
            rts -= dx;
        }
        if (@abs(dx) < tol) {
            return rts;
        }
        func_current = f(rts);
        diff_func_current = df(rts);
        if (func_current < 0.0) {
            xl = rts;
        } else {
            xh = rts;
        }
    }
    return error.NotConverged;
}

test "bisection finds root of x^2 - 4" {
    const f = struct {
        fn call(x: f64) f64 {
            return x * x - 4.0;
        }
    }.call;
    const root = try bisection(f, 0.0, 3.0, 1e-10, 100);
    try std.testing.expectApproxEqAbs(root, 2.0, 1e-6);
}

test "bisection finds cube root of 2" {
    const f = struct {
        fn call(x: f64) f64 {
            return x * x * x - 2.0;
        }
    }.call;
    const root = try bisection(f, 0.0, 2.0, 1e-10, 100);
    try std.testing.expectApproxEqAbs(root, std.math.cbrt(@as(f64, 2.0)), 1e-6);
}

test "brent finds root of x^2 - 4" {
    const f = struct {
        fn call(x: f64) f64 {
            return x * x - 4.0;
        }
    }.call;
    const root = try brent(f, 0.0, 3.0, 1e-10, 100);
    try std.testing.expectApproxEqAbs(root, 2.0, 1e-6);
}

test "newton finds root of x^2 - 4" {
    const f = struct {
        fn call(x: f64) f64 {
            return x * x - 4.0;
        }
    }.call;
    const df = struct {
        fn call(x: f64) f64 {
            return 2.0 * x;
        }
    }.call;
    const root = try newton(f, df, 3.0, 1e-10, 100);
    try std.testing.expectApproxEqAbs(root, 2.0, 1e-6);
}

test "newton_bisection finds root of x^2 - 4" {
    const f = struct {
        fn call(x: f64) f64 {
            return x * x - 4.0;
        }
    }.call;
    const df = struct {
        fn call(x: f64) f64 {
            return 2.0 * x;
        }
    }.call;
    const root = try newton_bisection(f, df, 0.0, 3.0, 1e-10, 100);
    try std.testing.expectApproxEqAbs(root, 2.0, 1e-6);
}

test "newton_bisection finds cube root of 2" {
    const f = struct {
        fn call(x: f64) f64 {
            return x * x * x - 2.0;
        }
    }.call;
    const df = struct {
        fn call(x: f64) f64 {
            return 3.0 * x * x;
        }
    }.call;
    const root = try newton_bisection(f, df, 0.0, 2.0, 1e-10, 100);
    try std.testing.expectApproxEqAbs(root, std.math.cbrt(@as(f64, 2.0)), 1e-6);
}

test "bisection returns NotConverged for non-bracketing interval" {
    const f = struct {
        fn call(x: f64) f64 {
            return x * x - 4.0;
        }
    }.call;
    const result = bisection(f, 3.0, 5.0, 1e-10, 100);
    try std.testing.expectError(error.NotConverged, result);
}
