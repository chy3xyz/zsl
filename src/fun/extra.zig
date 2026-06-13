const std = @import("std");
const gamma = @import("gamma.zig");
const float = @import("../float.zig");

/// Returns true when `x` is a finite, exact integer-valued `f64`.
fn isInteger(x: f64) bool {
    if (!std.math.isFinite(x)) return false;
    return @floor(x) == x;
}

/// Euler beta function: B(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b).
///
/// Uses the log-gamma implementation to avoid overflow.
pub fn beta(a: f64, b: f64) f64 {
    return @exp(gamma.log_gamma(a) + gamma.log_gamma(b) - gamma.log_gamma(a + b));
}

/// Generalized binomial coefficient.
///
/// For non-negative integer arguments with `k <= n`, an exact factorial
/// ratio is used when `n` is small. Otherwise the beta-function form is
/// used, which is valid for real `n` and `k`.
///
/// Returns `0` when `k < 0` or `k > n` for integer inputs.
pub fn binomial(n: f64, k: f64) f64 {
    if (isInteger(n) and isInteger(k)) {
        const ki = @as(i64, @intFromFloat(@floor(k)));
        const ni = @as(i64, @intFromFloat(@floor(n)));
        if (ki < 0 or ki > ni) return 0.0;
        if (ki == 0 or ki == ni) return 1.0;
        if (ki == 1 or ki == ni - 1) return @as(f64, @floatFromInt(ni));

        // Exact factorial ratio for small integer arguments.
        if (ni >= 0 and ni <= 22) {
            const un = @as(usize, @intCast(ni));
            const uk = @as(usize, @intCast(@min(ki, ni - ki)));
            return @floor(0.5 +
                gamma.factorial(un) / (gamma.factorial(uk) * gamma.factorial(un - uk)));
        }
    }

    // General real case via the beta function.
    return 1.0 / ((n + 1.0) * beta(n - k + 1.0, k + 1.0));
}

/// Standard logistic sigmoid: 1 / (1 + exp(-x)).
pub fn logistic(x: f64) f64 {
    return 1.0 / (1.0 + @exp(-x));
}

/// Rectified Linear Unit: max(0, x).
pub fn relu(x: f64) f64 {
    return @max(0.0, x);
}

/// Leaky ReLU.
pub fn leaky_relu(x: f64, alpha: f64) f64 {
    return if (x >= 0.0) x else alpha * x;
}

/// Exponential Linear Unit.
pub fn elu(x: f64, alpha: f64) f64 {
    return if (x >= 0.0) x else alpha * (@exp(x) - 1.0);
}

/// Scaled Exponential Linear Unit.
pub fn selu(x: f64) f64 {
    const scale: f64 = 1.0507009873554804934193349852946;
    const alpha: f64 = 1.6732632423543772848170429916717;
    return scale * if (x >= 0.0) x else alpha * (@exp(x) - 1.0);
}

/// Gaussian Error Linear Unit (approximate).
pub fn gelu(x: f64) f64 {
    const sqrt_two_over_pi: f64 = 0.79788456080286535587989211986876;
    const c: f64 = 0.044715;
    const arg = sqrt_two_over_pi * (x + c * x * x * x);
    return 0.5 * x * (1.0 + std.math.tanh(arg));
}

/// Alias for `logistic`.
pub fn sigmoid(x: f64) f64 {
    return logistic(x);
}

/// Hyperbolic tangent activation.
pub fn tanh_activation(x: f64) f64 {
    return std.math.tanh(x);
}

/// Softplus: ln(1 + exp(x)), computed with mild overflow/underflow care.
pub fn softplus(x: f64) f64 {
    if (x > 0.0) {
        return x + @log(1.0 + @exp(-x));
    }
    return @log(1.0 + @exp(x));
}

/// Softsign: x / (1 + |x|).
pub fn softsign(x: f64) f64 {
    return x / (1.0 + @abs(x));
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

test "beta reference values" {
    const eps = 1e-12;
    try std.testing.expect(float.approxEqAbs(f64, beta(0.5, 0.5), std.math.pi, eps));
    try std.testing.expect(float.approxEqAbs(f64, beta(1.0, 1.0), 1.0, eps));
}

test "binomial reference values" {
    try std.testing.expectEqual(@as(f64, 10.0), binomial(5.0, 2.0));
    try std.testing.expectEqual(@as(f64, 1.0), binomial(0.0, 0.0));
    try std.testing.expectEqual(@as(f64, 1.0), binomial(5.0, 0.0));
    try std.testing.expectEqual(@as(f64, 1.0), binomial(5.0, 5.0));
    try std.testing.expectEqual(@as(f64, 252.0), binomial(10.0, 5.0));
    try std.testing.expectEqual(@as(f64, 5.0), binomial(5.0, 1.0));
    try std.testing.expectEqual(@as(f64, 0.0), binomial(5.0, -1.0));
    try std.testing.expectEqual(@as(f64, 0.0), binomial(5.0, 6.0));
    try std.testing.expectEqual(@as(f64, 0.0), binomial(-1.0, 2.0));

    const eps = 1e-12;
    try std.testing.expect(float.approxEqAbs(f64, binomial(4.5, 2.0), 7.875, eps));
}

test "logistic and sigmoid" {
    try std.testing.expectEqual(@as(f64, 0.5), logistic(0.0));
    try std.testing.expectEqual(@as(f64, 0.5), sigmoid(0.0));

    const eps = 1e-12;
    try std.testing.expect(float.approxEqAbs(f64, logistic(1.0), 1.0 / (1.0 + @exp(-1.0)), eps));
}

test "relu" {
    try std.testing.expectEqual(@as(f64, 0.0), relu(-3.0));
    try std.testing.expectEqual(@as(f64, 4.0), relu(4.0));
}

test "leaky_relu" {
    const eps = 1e-15;
    try std.testing.expectEqual(@as(f64, 4.0), leaky_relu(4.0, 0.2));
    try std.testing.expect(float.approxEqAbs(f64, leaky_relu(-3.0, 0.2), -0.6, eps));
}

test "elu" {
    const alpha: f64 = 1.0;
    try std.testing.expectEqual(@as(f64, 4.0), elu(4.0, alpha));
    try std.testing.expect(float.approxEqAbs(f64, elu(-1.0, alpha), alpha * (@exp(-1.0) - 1.0), 1e-12));
}

test "selu" {
    const scale: f64 = 1.0507009873554804934193349852946;
    const alpha: f64 = 1.6732632423543772848170429916717;
    try std.testing.expectEqual(@as(f64, scale * 2.0), selu(2.0));
    try std.testing.expect(float.approxEqAbs(f64, selu(-1.0), scale * alpha * (@exp(-1.0) - 1.0), 1e-12));
}

test "gelu" {
    const eps = 1e-12;
    try std.testing.expect(float.approxEqAbs(f64, gelu(0.0), 0.0, eps));
    try std.testing.expect(float.approxEqAbs(f64, gelu(1.0), 0.8411919906082768, eps));
}

test "tanh_activation" {
    const eps = 1e-12;
    try std.testing.expect(float.approxEqAbs(f64, tanh_activation(0.0), 0.0, eps));
    try std.testing.expect(float.approxEqAbs(f64, tanh_activation(1.0), std.math.tanh(@as(f64, 1.0)), eps));
}

test "softplus" {
    const eps = 1e-12;
    try std.testing.expect(float.approxEqAbs(f64, softplus(0.0), @log(2.0), eps));
    try std.testing.expect(float.approxEqAbs(f64, softplus(2.0), @log(1.0 + @exp(2.0)), eps));
}

test "softsign" {
    const eps = 1e-12;
    try std.testing.expect(float.approxEqAbs(f64, softsign(0.0), 0.0, eps));
    try std.testing.expect(float.approxEqAbs(f64, softsign(2.0), 2.0 / 3.0, eps));
    try std.testing.expect(float.approxEqAbs(f64, softsign(-2.0), -2.0 / 3.0, eps));
}
