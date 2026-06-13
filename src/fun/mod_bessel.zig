const std = @import("std");
const tab = @import("mod_bessel_tables.zig");

/// Evaluate a polynomial P(x) = c[0] + c[1]*x + ... + c[n]*x^n
/// using Horner's method.
fn mbpoly(cof: []const f64, x: f64) f64 {
    var ans = cof[cof.len - 1];
    var i: usize = cof.len - 1;
    while (i > 0) {
        i -= 1;
        ans = cof[i] + x * ans;
    }
    return ans;
}

/// Returns the modified Bessel function I0(x) for any real x.
pub fn @"i0"(x: f64) f64 {
    const ax = @abs(x);
    if (ax < 15.0) { // Rational approximation.
        const y = x * x;
        return mbpoly(&tab.i0p, y) / mbpoly(&tab.i0q, 225.0 - y);
    }
    // Rational approximation with exp(x)/sqrt(x) factored out.
    const z = 1.0 - 15.0 / ax;
    return std.math.exp(ax) * mbpoly(&tab.i0pp, z) / (mbpoly(&tab.i0qq, z) * std.math.sqrt(ax));
}

/// Returns the modified Bessel function I1(x) for any real x.
pub fn @"i1"(x: f64) f64 {
    const ax = @abs(x);
    if (ax < 15.0) { // Rational approximation.
        const y = x * x;
        return x * mbpoly(&tab.i1p, y) / mbpoly(&tab.i1q, 225.0 - y);
    }
    // Rational approximation with exp(x)/sqrt(x) factored out.
    const z = 1.0 - 15.0 / ax;
    const ans = std.math.exp(ax) * mbpoly(&tab.i1pp, z) / (mbpoly(&tab.i1qq, z) * std.math.sqrt(ax));
    return if (x > 0.0) ans else -ans;
}

/// Returns the modified Bessel function In(x) for any real x and n >= 0.
pub fn in(n: i32, x: f64) f64 {
    if (n == 0) {
        return @"i0"(x);
    }
    if (n == 1) {
        return @"i1"(x);
    }
    if (x * x <= 8.0 * std.math.floatMin(f64)) {
        return 0.0;
    }
    const acc = 200.0; // acc determines accuracy.
    const iexp = 1024 / 2; // numeric_limits<double>::max_exponent/2
    const tox = 2.0 / @abs(x);
    var bip: f64 = 0.0;
    var bi: f64 = 1.0;
    var bim: f64 = 0.0;
    var ans: f64 = 0.0;
    const m = @as(i32, @intFromFloat(std.math.sqrt(acc * @as(f64, @floatFromInt(n)))));
    var j: i32 = 2 * (n + m);
    while (j > 0) : (j -= 1) { // Downward recurrence.
        bim = bip + @as(f64, @floatFromInt(j)) * tox * bi;
        bip = bi;
        bi = bim;
        const k = std.math.frexp(bi).exponent;
        if (k > iexp) { // Renormalize to prevent overflows.
            ans = std.math.ldexp(ans, -iexp);
            bi = std.math.ldexp(bi, -iexp);
            bip = std.math.ldexp(bip, -iexp);
        }
        if (j == n) {
            ans = bip;
        }
    }
    ans *= @"i0"(x) / bi; // Normalize with I0.
    if (x < 0.0 and (n & 1) != 0) { // n&1 != 0 => n is odd
        return -ans;
    }
    return ans;
}

/// Returns the modified Bessel function K0(x) for positive real x.
/// Special cases:
///   K0(0) = +inf
///   K0(x<0) = nan
pub fn k0(x: f64) f64 {
    if (x < 0.0) {
        return std.math.nan(f64);
    }
    if (x == 0.0) {
        return std.math.inf(f64);
    }
    if (x <= 1.0) { // Use two rational approximations.
        const z = x * x;
        const term = mbpoly(&tab.k0pi, z) * std.math.log(f64, std.math.e, x) / mbpoly(&tab.k0qi, 1.0 - z);
        return mbpoly(&tab.k0p, z) / mbpoly(&tab.k0q, 1.0 - z) - term;
    }
    // Rational approximation with exp(-x)/sqrt(x) factored out.
    const z = 1.0 / x;
    return std.math.exp(-x) * mbpoly(&tab.k0pp, z) / (mbpoly(&tab.k0qq, z) * std.math.sqrt(x));
}

/// Returns the modified Bessel function K1(x) for positive real x.
/// Special cases:
///   K1(0) = +inf
///   K1(x<0) = nan
pub fn k1(x: f64) f64 {
    if (x < 0.0) {
        return std.math.nan(f64);
    }
    if (x == 0.0) {
        return std.math.inf(f64);
    }
    if (x <= 1.0) { // Use two rational approximations.
        const z = x * x;
        const term = mbpoly(&tab.k1pi, z) * std.math.log(f64, std.math.e, x) / mbpoly(&tab.k1qi, 1.0 - z);
        return x * (mbpoly(&tab.k1p, z) / mbpoly(&tab.k1q, 1.0 - z) + term) + 1.0 / x;
    }
    // Rational approximation with exp(-x)/sqrt(x) factored out.
    const z = 1.0 / x;
    return std.math.exp(-x) * mbpoly(&tab.k1pp, z) / (mbpoly(&tab.k1qq, z) * std.math.sqrt(x));
}

/// Returns the modified Bessel function Kn(x) for positive x and n >= 0.
pub fn kn(n: i32, x: f64) f64 {
    if (n == 0) {
        return k0(x);
    }
    if (n == 1) {
        return k1(x);
    }
    if (x < 0.0) {
        return std.math.nan(f64);
    }
    if (x == 0.0) {
        return std.math.inf(f64);
    }
    const tox = 2.0 / x;
    var bkm = k0(x); // Upward recurrence for all x...
    var bk = k1(x);
    var bkp: f64 = 0.0;
    var j: i32 = 1;
    while (j < n) : (j += 1) {
        bkp = bkm + @as(f64, @floatFromInt(j)) * tox * bk;
        bkm = bk;
        bk = bkp;
    }
    return bk;
}

test "i0 known values" {
    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), @"i0"(0.0), eps);
    try std.testing.expectApproxEqAbs(@as(f64, 1.2660658777520084), @"i0"(1.0), eps);
    try std.testing.expectApproxEqAbs(@as(f64, 1.2660658777520084), @"i0"(-1.0), eps);
    try std.testing.expectApproxEqAbs(@as(f64, 2815.716628466254), @"i0"(10.0), eps);
}

test "i1 known values" {
    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), @"i1"(0.0), eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0.565159103992485), @"i1"(1.0), eps);
    try std.testing.expectApproxEqAbs(@as(f64, -0.565159103992485), @"i1"(-1.0), eps);
}

test "in known values" {
    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 1.2660658777520084), in(0, 1.0), eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0.565159103992485), in(1, 1.0), eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0.13574766976703833), in(2, 1.0), eps);
    try std.testing.expectApproxEqAbs(@as(f64, -0.565159103992485), in(1, -1.0), eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0.13574766976703833), in(2, -1.0), eps);
}

test "k0 known values" {
    const eps = 1e-10;
    try std.testing.expect(std.math.isInf(k0(0.0)));
    try std.testing.expect(std.math.isNan(k0(-1.0)));
    try std.testing.expectApproxEqAbs(@as(f64, 0.42102443824070834), k0(1.0), eps);
}

test "k1 known values" {
    const eps = 1e-10;
    try std.testing.expect(std.math.isInf(k1(0.0)));
    try std.testing.expect(std.math.isNan(k1(-1.0)));
    try std.testing.expectApproxEqAbs(@as(f64, 0.6019072301972346), k1(1.0), eps);
}

test "kn known values" {
    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 0.42102443824070834), kn(0, 1.0), eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6019072301972346), kn(1, 1.0), eps);
    try std.testing.expectApproxEqAbs(@as(f64, 1.624838898635177), kn(2, 1.0), eps);
}
