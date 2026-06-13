const std = @import("std");
const complex = @import("complex.zig");
const Complex = complex.Complex;

// Coefficients for the Stirling asymptotic expansion of log-Gamma.
const a = [_]f64{
    0.08333333333333333,
    -2.777777777777778e-3,
    7.936507936507937e-4,
    -5.952380952380952e-4,
    8.417508417508418e-4,
    -1.917526917526918e-3,
    6.410256410256410e-3,
    -0.02955065359477124,
    0.1796443723688307,
    -1.39243221690590,
};

fn is_non_positive_int(x: f64) bool {
    return x <= 0.0 and x == @floor(x);
}

/// Shifted asymptotic expansion valid for Re(w) >= 7.
fn clog_gamma_shifted(w: Complex(f64)) Complex(f64) {
    const half = Complex(f64).new(0.5, 0.0);
    const log_w = w.log();
    const term = w.sub(half).mul(log_w).sub(w).add(
        Complex(f64).new(0.5 * @log(2.0 * std.math.pi), 0.0),
    );

    const one = Complex(f64).new(1.0, 0.0);
    var w_pow = one.div(w); // w^(-1)
    var sum = Complex(f64).new(0.0, 0.0);
    var k: usize = 0;
    while (k < 10) : (k += 1) {
        sum = sum.add(w_pow.scale(a[k]));
        w_pow = w_pow.div(w.mul(w)); // next odd negative power
    }

    return term.add(sum);
}

/// Compute the complex log-Gamma function (principal branch).
///
/// For negative real arguments the result follows the principal branch;
/// the imaginary part is +π (approaching ±π near the branch cut), reflecting
/// that Gamma(z) is real and negative on the negative real axis.
pub fn clog_gamma(z: Complex(f64)) Complex(f64) {
    if (z.im == 0.0 and is_non_positive_int(z.re)) {
        return Complex(f64).new(std.math.inf(f64), 0.0);
    }

    if (z.re < 0.5) {
        const pi_z = z.scale(std.math.pi);
        const sin_pi_z = pi_z.sin();
        const log_sin = sin_pi_z.log();
        const log_pi = Complex(f64).new(@log(std.math.pi), 0.0);
        const one_minus_z = Complex(f64).new(1.0, 0.0).sub(z);
        return log_pi.sub(log_sin).sub(clog_gamma(one_minus_z));
    }

    const n: i32 = if (z.re < 7.0) @as(i32, @intFromFloat(@ceil(7.0 - z.re))) else 0;
    const w = z.add(Complex(f64).new(@floatFromInt(n), 0.0));

    var log_sum = Complex(f64).new(0.0, 0.0);
    var k: i32 = 0;
    while (k < n) : (k += 1) {
        const zk = z.add(Complex(f64).new(@floatFromInt(k), 0.0));
        log_sum = log_sum.add(zk.log());
    }

    return clog_gamma_shifted(w).sub(log_sum);
}

/// Compute the complex Gamma function.
pub fn cgamma(z: Complex(f64)) Complex(f64) {
    return clog_gamma(z).exp();
}

test "cgamma reference values" {
    const eps = 1e-10;
    const C = Complex(f64);

    const g1 = cgamma(C.new(1.0, 0.0));
    try std.testing.expectApproxEqAbs(g1.re, 1.0, eps);
    try std.testing.expectApproxEqAbs(g1.im, 0.0, eps);

    const g2 = cgamma(C.new(2.0, 0.0));
    try std.testing.expectApproxEqAbs(g2.re, 1.0, eps);
    try std.testing.expectApproxEqAbs(g2.im, 0.0, eps);

    const g_half = cgamma(C.new(0.5, 0.0));
    try std.testing.expectApproxEqAbs(g_half.re, std.math.sqrt(std.math.pi), eps);
    try std.testing.expectApproxEqAbs(g_half.im, 0.0, eps);

    const g_complex = cgamma(C.new(1.0, 2.0));
    try std.testing.expectApproxEqAbs(g_complex.re, 0.15190400267003606, 1e-10);
    try std.testing.expectApproxEqAbs(g_complex.im, 0.01980488016185497, 1e-10);
}

test "clog_gamma reference values" {
    const eps = 1e-10;
    const C = Complex(f64);

    const lg1 = clog_gamma(C.new(1.0, 0.0));
    try std.testing.expectApproxEqAbs(lg1.re, 0.0, eps);
    try std.testing.expectApproxEqAbs(lg1.im, 0.0, eps);

    const lg2 = clog_gamma(C.new(2.0, 0.0));
    try std.testing.expectApproxEqAbs(lg2.re, 0.0, eps);
    try std.testing.expectApproxEqAbs(lg2.im, 0.0, eps);

    const lg_half = clog_gamma(C.new(0.5, 0.0));
    try std.testing.expectApproxEqAbs(lg_half.re, 0.5 * @log(std.math.pi), eps);
    try std.testing.expectApproxEqAbs(lg_half.im, 0.0, eps);
}

test "cgamma edge cases" {
    const eps = 1e-10;
    const C = Complex(f64);

    const g_neg_half = cgamma(C.new(-0.5, 0.0));
    try std.testing.expectApproxEqAbs(g_neg_half.re, -2.0 * std.math.sqrt(std.math.pi), eps);
    try std.testing.expectApproxEqAbs(g_neg_half.im, 0.0, eps);

    const g_neg_1_5 = cgamma(C.new(-1.5, 0.0));
    try std.testing.expectApproxEqAbs(g_neg_1_5.re, 4.0 * std.math.sqrt(std.math.pi) / 3.0, eps);
    try std.testing.expectApproxEqAbs(g_neg_1_5.im, 0.0, eps);

    const g_i = cgamma(C.new(0.0, 1.0));
    try std.testing.expectApproxEqAbs(g_i.re, -0.154949828301811, eps);
    try std.testing.expectApproxEqAbs(g_i.im, -0.498015668118356, eps);

    const lg_2_5 = clog_gamma(C.new(2.5, 0.0));
    try std.testing.expectApproxEqAbs(lg_2_5.re, 0.284682870472919, eps);
    try std.testing.expectApproxEqAbs(lg_2_5.im, 0.0, eps);
}
