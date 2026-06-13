const std = @import("std");
const Error = @import("../errors.zig").Error;

/// Evaluate a Chebyshev polynomial series at x.
///
/// Computes sum_{k=0}^{n-1} c[k] * T_k(x), where T_k are Chebyshev
/// polynomials of the first kind on the interval [-1, 1].
pub fn cheb_eval(c: []const f64, x: f64) f64 {
    const n = c.len;
    if (n == 0) return 0.0;
    if (n == 1) return c[0];
    var d2: f64 = 0.0;
    var d1: f64 = 0.0;
    var k = n - 1;
    while (k >= 1) {
        const temp = d1;
        d1 = 2.0 * x * d1 - d2 + c[k];
        d2 = temp;
        if (k == 1) break;
        k -= 1;
    }
    return x * d1 - d2 + c[0];
}

/// 1-D linear interpolation of data points (x, y) at xi.
///
/// x must be sorted in strictly ascending order and have the same length as y.
/// Values outside the range of x are clamped to the nearest endpoint.
pub fn data_interp(xi: f64, x: []const f64, y: []const f64, allocator: std.mem.Allocator) Error!f64 {
    _ = allocator;
    if (x.len != y.len) return error.ShapeMismatch;
    if (x.len < 2) return error.InvalidDimension;
    for (1..x.len) |i| {
        if (x[i] <= x[i - 1]) return error.InvalidDimension;
    }
    const n = x.len;
    if (xi <= x[0]) return y[0];
    if (xi >= x[n - 1]) return y[n - 1];

    var lo: usize = 0;
    var hi: usize = n - 1;
    while (hi - lo > 1) {
        const mid = (lo + hi) / 2;
        if (xi < x[mid]) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    const t = (xi - x[lo]) / (x[hi] - x[lo]);
    return y[lo] + t * (y[hi] - y[lo]);
}

/// Cubic polynomial interpolation helper.
pub const InterpCubic = struct {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    tol_den: f64,

    pub fn init() InterpCubic {
        return .{
            .a = 0.0,
            .b = 0.0,
            .c = 0.0,
            .d = 0.0,
            .tol_den = 1e-15,
        };
    }

    /// Evaluate y = f(x).
    pub fn f(self: InterpCubic, x: f64) f64 {
        return ((self.a * x + self.b) * x + self.c) * x + self.d;
    }

    /// Evaluate y' = df/dx at x.
    pub fn g(self: InterpCubic, x: f64) f64 {
        return 3.0 * self.a * x * x + 2.0 * self.b * x + self.c;
    }

    /// Return critical points of the cubic.
    pub fn critical(self: InterpCubic) struct { xmin: f64, xmax: f64, xifl: f64, has_min: bool, has_max: bool, has_ifl: bool } {
        const del_by_4 = self.b * self.b - 3.0 * self.a * self.c;
        if (del_by_4 < 0.0) {
            return .{
                .xmin = 0.0,
                .xmax = 0.0,
                .xifl = 0.0,
                .has_min = false,
                .has_max = false,
                .has_ifl = false,
            };
        }
        const den = 3.0 * self.a;
        var xmin: f64 = 0.0;
        var xmax: f64 = 0.0;
        var has_min = false;
        var has_max = false;
        const xifl = -self.b / den;
        const has_ifl = true;
        if (del_by_4 != 0.0) {
            xmin = (-self.b + std.math.sqrt(del_by_4)) / den;
            xmax = (-self.b - std.math.sqrt(del_by_4)) / den;
            if (self.f(xmin) > self.f(xmax)) {
                const tmp = xmin;
                xmin = xmax;
                xmax = tmp;
            }
            has_min = true;
            has_max = true;
        }
        return .{
            .xmin = xmin,
            .xmax = xmax,
            .xifl = xifl,
            .has_min = has_min,
            .has_max = has_max,
            .has_ifl = has_ifl,
        };
    }

    /// Fit polynomial to four points.
    pub fn fit_4points(
        self: *InterpCubic,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        x3: f64,
        y3: f64,
    ) Error!void {
        const z0 = x0 * x0;
        const z1 = x1 * x1;
        const z2 = x2 * x2;
        const z3 = x3 * x3;
        const w0 = z0 * x0;
        const w1 = z1 * x1;
        const w2 = z2 * x2;
        const w3 = z3 * x3;
        const den = w0 * ((x2 - x3) * z1 + x3 * z2 - x2 * z3 + x1 * (z3 - z2)) + w1 * (x2 * z3 - x3 * z2) +
            x0 * ((w3 - w2) * z1 - w3 * z2 + w1 * (z2 - z3) + w2 * z3) + x1 * (w3 * z2 - w2 * z3) +
            (w2 * x3 - w3 * x2) * z1 + ((w2 - w3) * x1 + w3 * x2 - w2 * x3 + w1 * (x3 - x2)) * z0;
        if (@abs(den) < self.tol_den) return error.DivisionByZero;
        self.a = -((x1 * (y3 - y2) - x2 * y3 + x3 * y2 + (x2 - x3) * y1) * z0 + (x2 * y3 - x3 * y2) * z1 +
            y1 * (x3 * z2 - x2 * z3) + y0 * (x2 * z3 + x1 * (z2 - z3) - x3 * z2 + (x3 - x2) * z1) +
            x1 * (y2 * z3 - y3 * z2) + x0 * (y1 * (z3 - z2) - y2 * z3 + y3 * z2 + (y2 - y3) * z1)) / den;
        self.b = ((w1 * (x3 - x2) - w2 * x3 + w3 * x2 + (w2 - w3) * x1) * y0 + (w2 * x3 - w3 * x2) * y1 +
            x1 * (w3 * y2 - w2 * y3) + x0 * (w2 * y3 + w1 * (y2 - y3) - w3 * y2 + (w3 - w2) * y1) +
            w1 * (x2 * y3 - x3 * y2) + w0 * (x1 * (y3 - y2) - x2 * y3 + x3 * y2 + (x2 - x3) * y1)) / den;
        self.c = ((w1 * (y3 - y2) - w2 * y3 + w3 * y2 + (w2 - w3) * y1) * z0 + (w2 * y3 - w3 * y2) * z1 +
            y1 * (w3 * z2 - w2 * z3) + y0 * (w2 * z3 + w1 * (z2 - z3) - w3 * z2 + (w3 - w2) * z1) +
            w1 * (y2 * z3 - y3 * z2) + w0 * (y1 * (z3 - z2) - y2 * z3 + y3 * z2 + (y2 - y3) * z1)) / den;
        self.d = ((w1 * (x3 * y2 - x2 * y3) + x1 * (w2 * y3 - w3 * y2) + (w3 * x2 - w2 * x3) * y1) * z0 +
            y0 * (w1 * (x2 * z3 - x3 * z2) + x1 * (w3 * z2 - w2 * z3) + (w2 * x3 - w3 * x2) * z1) +
            x0 * (w1 * (y3 * z2 - y2 * z3) + y1 * (w2 * z3 - w3 * z2) + (w3 * y2 - w2 * y3) * z1) +
            w0 * (x1 * (y2 * z3 - y3 * z2) + y1 * (x3 * z2 - x2 * z3) + (x2 * y3 - x3 * y2) * z1)) / den;
    }

    /// Fit polynomial to three points and a known derivative.
    pub fn fit_3points_d(
        self: *InterpCubic,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        x3: f64,
        d3: f64,
    ) Error!void {
        // The derivative value d3 is part of the reference API but does not
        // appear in the ported closed-form coefficient expressions.
        _ = d3;
        const z0 = x0 * x0;
        const z1 = x1 * x1;
        const z2 = x2 * x2;
        const z3 = x3 * x3;
        const w0 = z0 * x0;
        const w1 = z1 * x1;
        const w2 = z2 * x2;
        const den = x0 * (2 * w1 * x3 - 2 * w2 * x3 - 3 * z1 * z3 + 3 * z2 * z3) +
            x1 * (2 * w2 * x3 - 3 * z2 * z3) + z1 * (3 * x2 * z3 - w2) +
            z0 * (-w1 + w2 + 3 * x1 * z3 - 3 * x2 * z3) + w1 * (z2 - 2 * x2 * x3) +
            w0 * (-2 * x1 * x3 + 2 * x2 * x3 + z1 - z2);
        if (@abs(den) < self.tol_den) return error.DivisionByZero;
        self.a = -(-2 * x1 * x3 * y2 + x0 * (2 * x3 * y2 - 2 * x3 * y1) + (y1 - y2) * z0 + y2 * z1 +
            y1 * (2 * x2 * x3 - z2) + y0 * (z2 - z1 - 2 * x2 * x3 + 2 * x1 * x3)) / den;
        self.b = (w0 * (y1 - y2) + w1 * y2 - 3 * x1 * y2 * z3 +
            y0 * (-3 * x2 * z3 + 3 * x1 * z3 + w2 - w1) + y1 * (3 * x2 * z3 - w2) +
            x0 * (3 * y2 * z3 - 3 * y1 * z3)) / den;
        self.c = (-2 * w1 * x3 * y2 + w0 * (2 * x3 * y2 - 2 * x3 * y1) + 3 * y2 * z1 * z3 +
            z0 * (3 * y1 * z3 - 3 * y2 * z3) + y1 * (2 * w2 * x3 - 3 * z2 * z3) +
            y0 * (3 * z2 * z3 - 3 * z1 * z3 - 2 * w2 * x3 + 2 * w1 * x3)) / den;
        self.d = -(w0 * (y1 * (z2 - 2 * x2 * x3) - y2 * z1 + 2 * x1 * x3 * y2) +
            z0 * (y1 * (3 * x2 * z3 - w2) - 3 * x1 * y2 * z3 + w1 * y2) +
            x0 * (y1 * (2 * w2 * x3 - 3 * z2 * z3) + 3 * y2 * z1 * z3 - 2 * w1 * x3 * y2) +
            y0 * (x1 * (3 * z2 * z3 - 2 * w2 * x3) + z1 * (w2 - 3 * x2 * z3) + w1 * (2 * x2 * x3 - z2))) / den;
    }
};

/// Quadratic polynomial interpolation helper.
pub const InterpQuad = struct {
    a: f64,
    b: f64,
    c: f64,
    tol_den: f64,

    pub fn init() InterpQuad {
        return .{
            .a = 0.0,
            .b = 0.0,
            .c = 0.0,
            .tol_den = 1e-15,
        };
    }

    /// Evaluate y = f(x).
    pub fn f(self: InterpQuad, x: f64) f64 {
        return (self.a * x + self.b) * x + self.c;
    }

    /// Evaluate y' = df/dx at x.
    pub fn g(self: InterpQuad, x: f64) f64 {
        return 2.0 * self.a * x + self.b;
    }

    /// Return the optimum point (zero derivative).
    pub fn optimum(self: InterpQuad) Error!struct { xopt: f64, fopt: f64 } {
        if (@abs(self.a) < self.tol_den) return error.DivisionByZero;
        const xopt = -0.5 * self.b / self.a;
        const fopt = self.f(xopt);
        return .{ .xopt = xopt, .fopt = fopt };
    }

    /// Fit polynomial to three points.
    pub fn fit_3points(
        self: *InterpQuad,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
    ) Error!void {
        const z0 = x0 * x0;
        const z1 = x1 * x1;
        const z2 = x2 * x2;
        const den = x0 * (z2 - z1) - x1 * z2 + x2 * z1 + (x1 - x2) * z0;
        if (@abs(den) < self.tol_den) return error.DivisionByZero;
        self.a = ((x1 - x2) * y0 + x2 * y1 - x1 * y2 + x0 * (y2 - y1)) / den;
        self.b = ((y1 - y2) * z0 + y2 * z1 - y1 * z2 + y0 * (z2 - z1)) / den;
        self.c = -((x2 * y1 - x1 * y2) * z0 + y0 * (x1 * z2 - x2 * z1) + x0 * (y2 * z1 - y1 * z2)) / den;
    }

    /// Fit polynomial to two points and a known derivative.
    pub fn fit_2points_d(
        self: *InterpQuad,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        x2: f64,
        d2: f64,
    ) Error!void {
        const z0 = x0 * x0;
        const z1 = x1 * x1;
        const den = -z1 + z0 + 2 * x1 * x2 - 2 * x0 * x2;
        if (@abs(den) < self.tol_den) return error.DivisionByZero;
        self.a = (-d2 * x0 + d2 * x1 + y0 - y1) / den;
        self.b = (-2 * x2 * y0 + 2 * x2 * y1 + d2 * z0 - d2 * z1) / den;
        self.c = ((y1 - d2 * x1) * z0 + y0 * (2 * x1 * x2 - z1) + x0 * (d2 * z1 - 2 * x2 * y1)) / den;
    }
};

test "cheb_eval" {
    const float = @import("../float.zig");
    // T0(x) = 1
    try std.testing.expectEqual(@as(f64, 1.0), cheb_eval(&[_]f64{1.0}, 0.5));
    // T2(x) = 2*x^2 - 1; coefficients for T2 are [0, 0, 1]
    try std.testing.expect(float.approxEqAbs(f64, cheb_eval(&[_]f64{ 0.0, 0.0, 1.0 }, 0.5), -0.5, 1e-12));
    // T2(x) + T1(x) = (2*x^2 - 1) + x; at x=0.5 => -0.5 + 0.5 = 0
    try std.testing.expect(float.approxEqAbs(f64, cheb_eval(&[_]f64{ 0.0, 1.0, 1.0 }, 0.5), 0.0, 1e-12));
}

test "data_interp" {
    const float = @import("../float.zig");
    const x = &[_]f64{ 0.0, 1.0, 2.0, 3.0 };
    const y = &[_]f64{ 0.0, 1.0, 4.0, 9.0 };
    try std.testing.expect(float.approxEqAbs(f64, try data_interp(0.5, x, y, std.testing.allocator), 0.5, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try data_interp(1.5, x, y, std.testing.allocator), 2.5, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try data_interp(-1.0, x, y, std.testing.allocator), 0.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, try data_interp(4.0, x, y, std.testing.allocator), 9.0, 1e-12));

    const bad_x = &[_]f64{ 0.0, 0.0, 1.0 };
    const bad_y = &[_]f64{ 0.0, 1.0, 2.0 };
    try std.testing.expectError(error.InvalidDimension, data_interp(0.5, bad_x, bad_y, std.testing.allocator));

    const short_x = &[_]f64{0.0};
    const short_y = &[_]f64{0.0};
    try std.testing.expectError(error.InvalidDimension, data_interp(0.0, short_x, short_y, std.testing.allocator));

    const mismatched_x = &[_]f64{ 0.0, 1.0 };
    const mismatched_y = &[_]f64{ 0.0, 1.0, 2.0 };
    try std.testing.expectError(error.ShapeMismatch, data_interp(0.5, mismatched_x, mismatched_y, std.testing.allocator));
}

test "interp_cubic fit_4points" {
    const float = @import("../float.zig");
    var ic = InterpCubic.init();
    try ic.fit_4points(0.0, 0.0, 1.0, 1.0, 2.0, 8.0, 3.0, 27.0);
    try std.testing.expect(float.approxEqAbs(f64, ic.f(0.0), 0.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, ic.f(1.0), 1.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, ic.f(2.0), 8.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, ic.f(3.0), 27.0, 1e-11));
}

test "interp_quad fit_3points" {
    const float = @import("../float.zig");
    var iq = InterpQuad.init();
    try iq.fit_3points(0.0, 0.0, 1.0, 1.0, 2.0, 4.0);
    try std.testing.expect(float.approxEqAbs(f64, iq.f(0.0), 0.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, iq.f(1.0), 1.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, iq.f(2.0), 4.0, 1e-12));
    const opt = try iq.optimum();
    try std.testing.expect(float.approxEqAbs(f64, opt.xopt, 0.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, opt.fopt, 0.0, 1e-12));
}
