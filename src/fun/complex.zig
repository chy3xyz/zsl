const std = @import("std");

/// A generic complex number with real and imaginary parts of type `T`.
pub fn Complex(comptime T: type) type {
    return struct {
        const Self = @This();

        re: T,
        im: T,

        /// Create a new complex number.
        pub fn new(re: T, im: T) Self {
            return .{ .re = re, .im = im };
        }

        /// Create a complex number from a real value.
        pub fn from_real(x: T) Self {
            return .{ .re = x, .im = 0 };
        }

        /// Add two complex numbers.
        pub fn add(self: Self, other: Self) Self {
            return .{ .re = self.re + other.re, .im = self.im + other.im };
        }

        /// Subtract two complex numbers.
        pub fn sub(self: Self, other: Self) Self {
            return .{ .re = self.re - other.re, .im = self.im - other.im };
        }

        /// Multiply two complex numbers.
        pub fn mul(self: Self, other: Self) Self {
            return .{
                .re = self.re * other.re - self.im * other.im,
                .im = self.re * other.im + self.im * other.re,
            };
        }

        /// Divide two complex numbers.
        pub fn div(self: Self, other: Self) Self {
            const denom = other.re * other.re + other.im * other.im;
            return .{
                .re = (self.re * other.re + self.im * other.im) / denom,
                .im = (self.im * other.re - self.re * other.im) / denom,
            };
        }

        /// Scale a complex number by a real factor.
        pub fn scale(self: Self, s: T) Self {
            return .{ .re = self.re * s, .im = self.im * s };
        }

        /// Return the complex conjugate.
        pub fn conj(self: Self) Self {
            return .{ .re = self.re, .im = -self.im };
        }

        /// Return the magnitude (absolute value).
        pub fn abs(self: Self) T {
            return std.math.hypot(self.re, self.im);
        }

        /// Return the argument (angle).
        pub fn arg(self: Self) T {
            return std.math.atan2(self.im, self.re);
        }

        /// Return the complex exponential.
        pub fn exp(self: Self) Self {
            const r = std.math.exp(self.re);
            return .{
                .re = r * std.math.cos(self.im),
                .im = r * std.math.sin(self.im),
            };
        }

        /// Return the principal complex logarithm.
        pub fn log(self: Self) Self {
            const r = self.abs();
            const theta = self.arg();
            return .{
                .re = @log(r),
                .im = theta,
            };
        }

        /// Return the principal complex square root.
        pub fn sqrt(self: Self) Self {
            return self.log().scale(0.5).exp();
        }

        /// Return the complex sine.
        pub fn sin(self: Self) Self {
            return .{
                .re = std.math.sin(self.re) * std.math.cosh(self.im),
                .im = std.math.cos(self.re) * std.math.sinh(self.im),
            };
        }

        /// Return the complex cosine.
        pub fn cos(self: Self) Self {
            return .{
                .re = std.math.cos(self.re) * std.math.cosh(self.im),
                .im = -std.math.sin(self.re) * std.math.sinh(self.im),
            };
        }

        /// Return the complex hyperbolic sine.
        pub fn sinh(self: Self) Self {
            return .{
                .re = std.math.sinh(self.re) * std.math.cos(self.im),
                .im = std.math.cosh(self.re) * std.math.sin(self.im),
            };
        }

        /// Return the complex hyperbolic cosine.
        pub fn cosh(self: Self) Self {
            return .{
                .re = std.math.cosh(self.re) * std.math.cos(self.im),
                .im = std.math.sinh(self.re) * std.math.sin(self.im),
            };
        }

        /// Raise a complex number to a complex power.
        ///
        /// Convention for 0^z:
        /// - 0^0 = 1
        /// - 0^z = complex infinity for Re(z) < 0
        /// - 0^z = 0         for Re(z) > 0
        pub fn pow(self: Self, exponent: Self) Self {
            if (self.re == 0 and self.im == 0) {
                if (exponent.re == 0 and exponent.im == 0) {
                    return Self.new(1, 0);
                }
                if (exponent.re < 0) {
                    return Self.new(std.math.inf(T), std.math.inf(T));
                }
                return Self.new(0, 0);
            }
            return self.log().mul(exponent).exp();
        }
    };
}

test "complex arithmetic" {
    const C = Complex(f64);
    const z1 = C.new(1.0, 2.0);
    const z2 = C.new(3.0, -1.0);

    const sum = z1.add(z2);
    try std.testing.expectApproxEqAbs(sum.re, 4.0, 1e-12);
    try std.testing.expectApproxEqAbs(sum.im, 1.0, 1e-12);

    const diff = z1.sub(z2);
    try std.testing.expectApproxEqAbs(diff.re, -2.0, 1e-12);
    try std.testing.expectApproxEqAbs(diff.im, 3.0, 1e-12);

    const prod = z1.mul(z2);
    try std.testing.expectApproxEqAbs(prod.re, 5.0, 1e-12);
    try std.testing.expectApproxEqAbs(prod.im, 5.0, 1e-12);

    const quot = z1.div(z2);
    try std.testing.expectApproxEqAbs(quot.re, 0.1, 1e-12);
    try std.testing.expectApproxEqAbs(quot.im, 0.7, 1e-12);

    const scaled = z1.scale(2.0);
    try std.testing.expectApproxEqAbs(scaled.re, 2.0, 1e-12);
    try std.testing.expectApproxEqAbs(scaled.im, 4.0, 1e-12);

    const conjugate = z1.conj();
    try std.testing.expectApproxEqAbs(conjugate.re, 1.0, 1e-12);
    try std.testing.expectApproxEqAbs(conjugate.im, -2.0, 1e-12);

    try std.testing.expectApproxEqAbs(z1.abs(), std.math.sqrt(5.0), 1e-12);
    try std.testing.expectApproxEqAbs(z1.arg(), std.math.atan2(@as(f64, 2.0), @as(f64, 1.0)), 1e-12);

    const e = z1.exp();
    const expected_exp = C.new(std.math.exp(1.0) * std.math.cos(2.0), std.math.exp(1.0) * std.math.sin(2.0));
    try std.testing.expectApproxEqAbs(e.re, expected_exp.re, 1e-12);
    try std.testing.expectApproxEqAbs(e.im, expected_exp.im, 1e-12);
}

test "complex edge cases" {
    const C = Complex(f64);
    const eps = 1e-12;

    const sqrt_neg1 = C.new(-1.0, 0.0).sqrt();
    try std.testing.expectApproxEqAbs(sqrt_neg1.re, 0.0, eps);
    try std.testing.expectApproxEqAbs(sqrt_neg1.im, 1.0, eps);

    const sqrt_neg4 = C.new(-4.0, 0.0).sqrt();
    try std.testing.expectApproxEqAbs(sqrt_neg4.re, 0.0, eps);
    try std.testing.expectApproxEqAbs(sqrt_neg4.im, 2.0, eps);

    const log_neg2 = C.new(-2.0, 0.0).log();
    try std.testing.expectApproxEqAbs(log_neg2.re, @log(2.0), eps);
    try std.testing.expectApproxEqAbs(log_neg2.im, std.math.pi, eps);

    const pow_neg1_half = C.new(-1.0, 0.0).pow(C.new(0.5, 0.0));
    try std.testing.expectApproxEqAbs(pow_neg1_half.re, 0.0, eps);
    try std.testing.expectApproxEqAbs(pow_neg1_half.im, 1.0, eps);

    const zero_to_zero = C.new(0.0, 0.0).pow(C.new(0.0, 0.0));
    try std.testing.expectApproxEqAbs(zero_to_zero.re, 1.0, eps);
    try std.testing.expectApproxEqAbs(zero_to_zero.im, 0.0, eps);

    const zero_to_pos = C.new(0.0, 0.0).pow(C.new(2.0, 0.0));
    try std.testing.expectApproxEqAbs(zero_to_pos.re, 0.0, eps);
    try std.testing.expectApproxEqAbs(zero_to_pos.im, 0.0, eps);

    const zero_to_neg = C.new(0.0, 0.0).pow(C.new(-1.0, 0.0));
    try std.testing.expect(std.math.isPositiveInf(zero_to_neg.re));
    try std.testing.expect(std.math.isPositiveInf(zero_to_neg.im));
}
