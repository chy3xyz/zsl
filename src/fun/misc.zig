const std = @import("std");
const Error = @import("../errors.zig").Error;

/// Compute the binomial coefficient n choose k.
/// Returns `error.InvalidDimension` for negative n or k, or when k > n.
pub fn choose(n: i64, k: i64) Error!f64 {
    if (n < 0 or k < 0 or n - k < 0) return error.InvalidDimension;
    const kk: i64 = @min(k, n - k);
    var result: f64 = 1.0;
    var i: i64 = 0;
    while (i < kk) : (i += 1) {
        result = result * @as(f64, @floatFromInt(n - i)) / @as(f64, @floatFromInt(i + 1));
    }
    return result;
}

/// Return the nth Fibonacci number.
/// Returns `error.InvalidDimension` for negative n.
pub fn fib(n: i64) Error!i64 {
    if (n < 0) return error.InvalidDimension;
    if (n == 0) return 0;
    if (n == 1) return 1;
    var a: i64 = 0;
    var b: i64 = 1;
    var i: i64 = 2;
    while (i <= n) : (i += 1) {
        const c = a + b;
        a = b;
        b = c;
    }
    return b;
}

/// Return sqrt(x^2 + y^2) computed without undue overflow/underflow.
pub fn hypot(x: f64, y: f64) f64 {
    return std.math.hypot(x, y);
}

/// Sinusoid representing the equation
///
///   y(t) = a0 + c1 * cos(omega_0 * t + theta)             [essential form]
///   y(t) = a0 + a1 * cos(omega_0 * t) + b1 * sin(omega_0 * t)  [basis form]
///
/// where
///   a1 =  c1 * cos(theta)
///   b1 = -c1 * sin(theta)
///   theta  = atan2(-b1, a1)
///   c1 = sqrt(a1^2 + b1^2)
pub const Sinusoid = struct {
    period: f64,
    mean_value: f64,
    amplitude: f64,
    phase_shift: f64,
    frequency: f64,
    angular_freq: f64,
    time_shift: f64,
    a: [2]f64,
    b: [2]f64,

    /// Create a sinusoid from essential parameters.
    ///   t     -- period
    ///   a0    -- mean value
    ///   c1    -- amplitude
    ///   theta -- phase shift (rad)
    pub fn essential(t: f64, a0: f64, c1: f64, theta: f64) Sinusoid {
        const frequency = 1.0 / t;
        const angular_freq = 2.0 * std.math.pi * frequency;
        const a1 = c1 * @cos(theta);
        const b1 = -c1 * @sin(theta);
        return .{
            .period = t,
            .mean_value = a0,
            .amplitude = c1,
            .phase_shift = theta,
            .frequency = frequency,
            .angular_freq = angular_freq,
            .time_shift = theta / angular_freq,
            .a = .{ a0, a1 },
            .b = .{ 0.0, b1 },
        };
    }

    /// Create a sinusoid from basis coefficients.
    ///   t  -- period
    ///   a0 -- mean value
    ///   a1 -- coefficient of cos term
    ///   b1 -- coefficient of sin term
    pub fn basis(t: f64, a0: f64, a1: f64, b1: f64) Sinusoid {
        const c1 = std.math.sqrt(a1 * a1 + b1 * b1);
        const theta = std.math.atan2(-b1, a1);
        const frequency = 1.0 / t;
        const angular_freq = 2.0 * std.math.pi * frequency;
        return .{
            .period = t,
            .mean_value = a0,
            .amplitude = c1,
            .phase_shift = theta,
            .frequency = frequency,
            .angular_freq = angular_freq,
            .time_shift = theta / angular_freq,
            .a = .{ a0, a1 },
            .b = .{ 0.0, b1 },
        };
    }

    /// Evaluate using the essential form.
    pub fn yessen(self: Sinusoid, t: f64) f64 {
        return self.mean_value + self.amplitude * @cos(self.angular_freq * t + self.phase_shift);
    }

    /// Evaluate using the basis form.
    pub fn ybasis(self: Sinusoid, t: f64) f64 {
        const omega_0 = self.angular_freq;
        var res = self.a[0];
        var i: usize = 1;
        while (i < self.a.len) : (i += 1) {
            const k = @as(f64, @floatFromInt(i));
            res += self.a[i] * @cos(k * omega_0 * t) + self.b[i] * @sin(k * omega_0 * t);
        }
        return res;
    }
};

test "choose" {
    try std.testing.expectEqual(@as(f64, 10.0), try choose(5, 2));
    try std.testing.expectEqual(@as(f64, 1.0), try choose(0, 0));
    try std.testing.expectEqual(@as(f64, 252.0), try choose(10, 5));
    try std.testing.expectError(error.InvalidDimension, choose(-1, 2));
    try std.testing.expectError(error.InvalidDimension, choose(5, -1));
    try std.testing.expectError(error.InvalidDimension, choose(5, 6));
}

test "fib" {
    try std.testing.expectEqual(@as(i64, 0), try fib(0));
    try std.testing.expectEqual(@as(i64, 1), try fib(1));
    try std.testing.expectEqual(@as(i64, 55), try fib(10));
    try std.testing.expectEqual(@as(i64, 6765), try fib(20));
    try std.testing.expectError(error.InvalidDimension, fib(-1));
}

test "hypot" {
    const float = @import("../float.zig");
    try std.testing.expect(float.approxEqAbs(f64, hypot(3.0, 4.0), 5.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, hypot(1e150, 1e150), std.math.sqrt2 * 1e150, 1e135));
    try std.testing.expectEqual(@as(f64, 0.0), hypot(0.0, 0.0));
}

test "sinusoid forms are equivalent" {
    const float = @import("../float.zig");
    const s = Sinusoid.essential(2.0, 1.0, 3.0, 0.5);
    try std.testing.expect(float.approxEqAbs(f64, s.yessen(0.25), s.ybasis(0.25), 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, s.yessen(0.0), s.ybasis(0.0), 1e-12));
}
