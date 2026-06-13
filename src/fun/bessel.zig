//! Bessel functions of the first and second kind.
//! Ported from VSL's `_ref/vsl/fun/bessel.v`.

const std = @import("std");
const tab = @import("bessel_tables.zig");

const two_m27 = std.math.pow(f64, 2.0, -27.0); // 2**-27
const two_m13 = std.math.pow(f64, 2.0, -13.0); // 2**-13
const two_m29 = std.math.pow(f64, 2.0, -29.0); // 2**-29
const two_m54 = std.math.pow(f64, 2.0, -54.0); // 2**-54

// j0r0/j0s0 on [0, 2]
const j0r02 = 1.56249999999999947958e-02;
const j0r03 = -1.89979294238854721751e-04;
const j0r04 = 1.82954049532700665670e-06;
const j0r05 = -4.61832688532103189199e-09;
const j0s01 = 1.56191029464890010492e-02;
const j0s02 = 1.16926784663337450260e-04;
const j0s03 = 5.13546550207318111446e-07;
const j0s04 = 1.16614003333790000205e-09;

const j1r00 = -6.25000000000000000000e-02;
const j1r01 = 1.40705666955189706048e-03;
const j1r02 = -1.59955631084035597520e-05;
const j1r03 = 4.96727999609584448412e-08;
const j1s01 = 1.91537599538363460805e-02;
const j1s02 = 1.85946785588630915560e-04;
const j1s03 = 1.17718464042623683263e-06;
const j1s04 = 5.04636257076217042715e-09;
const j1s05 = 1.23542274426137913908e-11;

const y0u00 = -7.38042951086872317523e-02;
const y0u01 = 1.76666452509181115538e-01;
const y0u02 = -1.38185671945596898896e-02;
const y0u03 = 3.47453432093683650238e-04;
const y0u04 = -3.81407053724364161125e-06;
const y0u05 = 1.95590137035022920206e-08;
const y0u06 = -3.98205194132103398453e-11;
const y0v01 = 1.27304834834123699328e-02;
const y0v02 = 7.60068627350353253702e-05;
const y0v03 = 2.59150851840457805467e-07;
const y0v04 = 4.41110311332675467403e-10;

const y1u00 = -1.96057090646238940668e-01;
const y1u01 = 5.04438716639811282616e-02;
const y1u02 = -1.91256895875763547298e-03;
const y1u03 = 2.35252600561610495928e-05;
const y1u04 = -9.19099158039878874504e-08;
const y1v00 = 1.99167318236649903973e-02;
const y1v01 = 2.02552581025135171496e-04;
const y1v02 = 1.35608801097516229404e-06;
const y1v03 = 6.22741452364621501295e-09;
const y1v04 = 1.66559246207992079114e-11;

const inv_sqrt_pi = 1.0 / std.math.sqrt(std.math.pi);

/// Returns the order-zero Bessel function of the first kind.
///
/// Special cases:
/// - `j0(±inf) = 0`
/// - `j0(0) = 1`
/// - `j0(nan) = nan`
pub fn j0(x_: f64) f64 {
    var x = x_;
    if (std.math.isNan(x)) {
        return x;
    }
    if (std.math.isInf(x)) {
        return 0.0;
    }
    if (x == 0.0) {
        return 1.0;
    }
    x = @abs(x);
    if (x >= 2.0) {
        const s = std.math.sin(x);
        const c = std.math.cos(x);
        var ss = s - c;
        var cc = s + c;
        // Make sure x+x does not overflow.
        if (x < std.math.floatMax(f64) / 2.0) {
            const z = -std.math.cos(x + x);
            if (s * c < 0.0) {
                cc = z / ss;
            } else {
                ss = z / cc;
            }
        }
        const u = pzero(x);
        const v = qzero(x);
        const z = inv_sqrt_pi * (u * cc - v * ss) / std.math.sqrt(x);
        return z;
    }
    if (x < two_m13) {
        if (x < two_m27) {
            return 1.0;
        }
        return 1.0 - 0.25 * x * x;
    }
    const z = x * x;
    const r = z * (j0r02 + z * (j0r03 + z * (j0r04 + z * j0r05)));
    const s = 1.0 + z * (j0s01 + z * (j0s02 + z * (j0s03 + z * j0s04)));
    if (x < 1.0) {
        return 1.0 + z * (-0.25 + (r / s));
    }
    const u = 0.5 * x;
    return (1.0 + u) * (1.0 - u) + z * (r / s);
}

/// Returns the order-one Bessel function of the first kind.
///
/// Special cases:
/// - `j1(±inf) = 0`
/// - `j1(nan) = nan`
pub fn j1(x_: f64) f64 {
    var x = x_;
    if (std.math.isNan(x)) {
        return x;
    }
    if (std.math.isInf(x) or x == 0.0) {
        return 0.0;
    }
    var sign = false;
    if (x < 0.0) {
        x = -x;
        sign = true;
    }
    if (x >= 2.0) {
        const s = std.math.sin(x);
        const c = std.math.cos(x);
        var ss = -s - c;
        var cc = s - c;
        if (x < std.math.floatMax(f64) / 2.0) {
            const z = std.math.cos(x + x);
            if (s * c > 0.0) {
                cc = z / ss;
            } else {
                ss = z / cc;
            }
        }
        const u = pone(x);
        const v = qone(x);
        const z = inv_sqrt_pi * (u * cc - v * ss) / std.math.sqrt(x);
        if (sign) {
            return -z;
        }
        return z;
    }
    if (x < two_m27) {
        return 0.5 * x;
    }
    const z = x * x;
    var r = z * (j1r00 + z * (j1r01 + z * (j1r02 + z * j1r03)));
    const s = 1.0 + z * (j1s01 + z * (j1s02 + z * (j1s03 + z * (j1s04 + z * j1s05))));
    r *= x;
    const result = 0.5 * x + r / s;
    if (sign) {
        return -result;
    }
    return result;
}

/// Returns the order-n Bessel function of the first kind.
///
/// Special cases:
/// - `jn(n, ±inf) = 0`
/// - `jn(n, nan) = nan`
pub fn jn(n_: i32, x_: f64) f64 {
    var n = n_;
    var x = x_;
    if (std.math.isNan(x)) {
        return x;
    }
    if (std.math.isInf(x)) {
        return 0.0;
    }
    // J(-n, x) = (-1)**n * J(n, x), J(n, -x) = (-1)**n * J(n, x)
    // Thus, J(-n, x) = J(n, -x)
    if (n == 0) {
        return j0(x);
    }
    if (x == 0.0) {
        return 0.0;
    }
    if (n < 0) {
        n = -n;
        x = -x;
    }
    if (n == 1) {
        return j1(x);
    }
    var sign = false;
    if (x < 0.0) {
        x = -x;
        if (n & 1 == 1) {
            sign = true; // odd n and negative x
        }
    }
    var b: f64 = 0.0;
    if (@as(f64, @floatFromInt(n)) <= x) {
        // Safe to use J(n+1,x) = 2n/x * J(n,x) - J(n-1,x)
        b = j1(x);
        var i: i32 = 1;
        var j0_x = j0(x);
        while (i < n) : (i += 1) {
            const prev_j0_x = j0_x;
            j0_x = b;
            b = b * (@as(f64, @floatFromInt(i)) * 2.0 / x) - prev_j0_x;
        }
    } else {
        if (x < two_m29) {
            // x is tiny, return the first Taylor expansion of J(n,x)
            // J(n,x) = 1/n! * (x/2)**n - ...
            if (n > 33) {
                b = 0.0;
            } else {
                const temp = x * 0.5;
                b = temp;
                var a: f64 = 1.0;
                var i: i32 = 2;
                while (i <= n) : (i += 1) {
                    a *= @as(f64, @floatFromInt(i)); // a = n!
                    b *= temp; // b = (x/2)**n
                }
                b /= a;
            }
        } else {
            // Use backward recurrence.
            const w = @as(f64, @floatFromInt(n)) * 2.0 / x;
            const h = 2.0 / x;
            var q0_ = w;
            var z = w + h;
            var q1_ = w * z - 1.0;
            var k: i32 = 1;
            while (q1_ < 1.0e+9) : (k += 1) {
                z += h;
                const q0_prev = q0_;
                q0_ = q1_;
                q1_ = z * q1_ - q0_prev;
            }
            const m = @as(i64, n) + @as(i64, n);
            var t: f64 = 0.0;
            var i: i64 = 2 * (@as(i64, n) + k);
            while (i >= m) : (i -= 2) {
                t = 1.0 / (@as(f64, @floatFromInt(i)) / x - t);
            }
            var t_ = t;
            b = 1.0;
            // Estimate log((2/x)**n * n!) = n*log(2/x) + n*ln(n).
            // Hence, if n*(log(2n/x)) > 7.09782712893383973096e+02
            // then recurrent value may overflow and the result is
            // likely underflow to zero.
            var tmp = @as(f64, @floatFromInt(n));
            const v = 2.0 / x;
            tmp = tmp * @log(@abs(v * tmp));
            if (tmp < 7.09782712893383973096e+02) {
                i = n - 1;
                while (i > 0) : (i -= 1) {
                    const di = @as(f64, @floatFromInt(i)) * 2.0;
                    const a = t_;
                    t_ = b;
                    b = b * di / x - a;
                }
            } else {
                i = n - 1;
                while (i > 0) : (i -= 1) {
                    const di = @as(f64, @floatFromInt(i)) * 2.0;
                    const a = t_;
                    t_ = b;
                    b = b * di / x - a;
                    // scale b to avoid spurious overflow
                    if (b > 1.0e+100) {
                        t_ /= b;
                        t /= b;
                        b = 1.0;
                    }
                }
            }
            b = t * j0(x) / b;
        }
    }
    if (sign) {
        return -b;
    }
    return b;
}

/// Returns the order-zero Bessel function of the second kind.
///
/// Special cases:
/// - `y0(+inf) = 0`
/// - `y0(0) = -inf`
/// - `y0(x < 0) = nan`
/// - `y0(nan) = nan`
pub fn y0(x: f64) f64 {
    if (x < 0.0 or std.math.isNan(x)) {
        return std.math.nan(f64);
    }
    if (std.math.isPositiveInf(x)) {
        return 0.0;
    }
    if (x == 0.0) {
        return -std.math.inf(f64);
    }
    if (x >= 2.0) {
        const s = std.math.sin(x);
        const c = std.math.cos(x);
        var ss = s - c;
        var cc = s + c;
        if (x < std.math.floatMax(f64) / 2.0) {
            const z = -std.math.cos(x + x);
            if (s * c < 0.0) {
                cc = z / ss;
            } else {
                ss = z / cc;
            }
        }
        const u = pzero(x);
        const v = qzero(x);
        const z = inv_sqrt_pi * (u * ss + v * cc) / std.math.sqrt(x);
        return z;
    }
    if (x <= two_m27) {
        return y0u00 + (2.0 / std.math.pi) * @log(x);
    }
    const z = x * x;
    const u = y0u00 + z * (y0u01 + z * (y0u02 + z * (y0u03 + z * (y0u04 + z * (y0u05 + z * y0u06)))));
    const v = 1.0 + z * (y0v01 + z * (y0v02 + z * (y0v03 + z * y0v04)));
    return u / v + (2.0 / std.math.pi) * j0(x) * @log(x);
}

/// Returns the order-one Bessel function of the second kind.
///
/// Special cases:
/// - `y1(+inf) = 0`
/// - `y1(0) = -inf`
/// - `y1(x < 0) = nan`
/// - `y1(nan) = nan`
pub fn y1(x: f64) f64 {
    if (x < 0.0 or std.math.isNan(x)) {
        return std.math.nan(f64);
    }
    if (std.math.isPositiveInf(x)) {
        return 0.0;
    }
    if (x == 0.0) {
        return -std.math.inf(f64);
    }
    if (x >= 2.0) {
        const s = std.math.sin(x);
        const c = std.math.cos(x);
        var ss = -s - c;
        var cc = s - c;
        if (x < std.math.floatMax(f64) / 2.0) {
            const z = std.math.cos(x + x);
            if (s * c > 0.0) {
                cc = z / ss;
            } else {
                ss = z / cc;
            }
        }
        const u = pone(x);
        const v = qone(x);
        const z = inv_sqrt_pi * (u * ss + v * cc) / std.math.sqrt(x);
        return z;
    }
    if (x <= two_m54) {
        return -(2.0 / std.math.pi) / x;
    }
    const z = x * x;
    const u = y1u00 + z * (y1u01 + z * (y1u02 + z * (y1u03 + z * y1u04)));
    const v = 1.0 + z * (y1v00 + z * (y1v01 + z * (y1v02 + z * (y1v03 + z * y1v04))));
    return x * (u / v) + (2.0 / std.math.pi) * (j1(x) * @log(x) - 1.0 / x);
}

/// Returns the order-n Bessel function of the second kind.
///
/// Special cases:
/// - `yn(n, +inf) = 0`
/// - `yn(n ≥ 0, 0) = -inf`
/// - `yn(n < 0, 0) = +inf if n is odd, -inf if n is even`
/// - `yn(n, x < 0) = nan`
/// - `yn(n, nan) = nan`
pub fn yn(n_: i32, x: f64) f64 {
    var n = n_;
    if (x < 0.0 or std.math.isNan(x)) {
        return std.math.nan(f64);
    }
    if (std.math.isPositiveInf(x)) {
        return 0.0;
    }
    if (n == 0) {
        return y0(x);
    }
    if (x == 0.0) {
        if (n < 0 and n & 1 == 1) {
            return std.math.inf(f64);
        }
        return -std.math.inf(f64);
    }
    var sign = false;
    if (n < 0) {
        n = -n;
        if (n & 1 == 1) {
            sign = true; // sign true if n < 0 and |n| odd
        }
    }
    if (n == 1) {
        if (sign) {
            return -y1(x);
        }
        return y1(x);
    }
    var y0_x = y0(x);
    var b = y1(x);
    // quit if b is -inf
    var i: i32 = 1;
    while (i < n and !std.math.isNegativeInf(b)) : (i += 1) {
        const prev_y0_x = y0_x;
        y0_x = b;
        b = (@as(f64, @floatFromInt(i)) * 2.0 / x) * b - prev_y0_x;
    }
    if (sign) {
        return -b;
    }
    return b;
}

// The asymptotic expansions of pzero is
// 1 - 9/128 s**2 + 11025/98304 s**4 - ..., where s = 1/x.
// For x >= 2, we approximate pzero by
// pzero(x) = 1 + (R/S)
// where R = pj0r0 + pR1*s**2 + pR2*s**4 + ... + pR5*s**10
// S = 1 + pj0s0*s**2 + ... + pS4*s**10
fn pzero(x: f64) f64 {
    const p: *const [6]f64 = if (x >= 8.0)
        &tab.p0r8
    else if (x >= 4.5454)
        &tab.p0r5
    else if (x >= 2.8571)
        &tab.p0r3
    else
        &tab.p0r2;
    const q: *const [5]f64 = if (x >= 8.0)
        &tab.p0s8
    else if (x >= 4.5454)
        &tab.p0s5
    else if (x >= 2.8571)
        &tab.p0s3
    else
        &tab.p0s2;
    const z = 1.0 / (x * x);
    const r = p[0] + z * (p[1] + z * (p[2] + z * (p[3] + z * (p[4] + z * p[5]))));
    const s = 1.0 + z * (q[0] + z * (q[1] + z * (q[2] + z * (q[3] + z * q[4]))));
    return 1.0 + r / s;
}

// For x >= 8, the asymptotic expansions of pone is
// 1 + 15/128 s**2 - 4725/2**15 s**4 - ..., where s = 1/x.
fn pone(x: f64) f64 {
    const p: *const [6]f64 = if (x >= 8.0)
        &tab.p1r8
    else if (x >= 4.5454)
        &tab.p1r5
    else if (x >= 2.8571)
        &tab.p1r3
    else
        &tab.p1r2;
    const q: *const [5]f64 = if (x >= 8.0)
        &tab.p1s8
    else if (x >= 4.5454)
        &tab.p1s5
    else if (x >= 2.8571)
        &tab.p1s3
    else
        &tab.p1s2;
    const z = 1.0 / (x * x);
    const r = p[0] + z * (p[1] + z * (p[2] + z * (p[3] + z * (p[4] + z * p[5]))));
    const s = 1.0 + z * (q[0] + z * (q[1] + z * (q[2] + z * (q[3] + z * q[4]))));
    return 1.0 + r / s;
}

// For x >= 8, the asymptotic expansions of qzero is
// -1/8 s + 75/1024 s**3 - ..., where s = 1/x.
fn qzero(x: f64) f64 {
    const p: *const [6]f64 = if (x >= 8.0)
        &tab.q0r8
    else if (x >= 4.5454)
        &tab.q0r5
    else if (x >= 2.8571)
        &tab.q0r3
    else
        &tab.q0r2;
    const q: *const [6]f64 = if (x >= 8.0)
        &tab.q0s8
    else if (x >= 4.5454)
        &tab.q0s5
    else if (x >= 2.8571)
        &tab.q0s3
    else
        &tab.q0s2;
    const z = 1.0 / (x * x);
    const r = p[0] + z * (p[1] + z * (p[2] + z * (p[3] + z * (p[4] + z * p[5]))));
    const s = 1.0 + z * (q[0] + z * (q[1] + z * (q[2] + z * (q[3] + z * (q[4] + z * q[5])))));
    return (-0.125 + r / s) / x;
}

// For x >= 8, the asymptotic expansions of qone is
// 3/8 s - 105/1024 s**3 - ..., where s = 1/x.
fn qone(x: f64) f64 {
    const p: *const [6]f64 = if (x >= 8.0)
        &tab.q1r8
    else if (x >= 4.5454)
        &tab.q1r5
    else if (x >= 2.8571)
        &tab.q1r3
    else
        &tab.q1r2;
    const q: *const [6]f64 = if (x >= 8.0)
        &tab.q1s8
    else if (x >= 4.5454)
        &tab.q1s5
    else if (x >= 2.8571)
        &tab.q1s3
    else
        &tab.q1s2;
    const z = 1.0 / (x * x);
    const r = p[0] + z * (p[1] + z * (p[2] + z * (p[3] + z * (p[4] + z * p[5]))));
    const s = 1.0 + z * (q[0] + z * (q[1] + z * (q[2] + z * (q[3] + z * (q[4] + z * q[5])))));
    return (0.375 + r / s) / x;
}

test "j0 special and known values" {
    const eps = 1.0e-10;
    try std.testing.expectEqual(1.0, j0(0.0));
    try std.testing.expectEqual(0.0, j0(std.math.inf(f64)));
    try std.testing.expectEqual(0.0, j0(-std.math.inf(f64)));
    try std.testing.expect(std.math.isNan(j0(std.math.nan(f64))));
    try std.testing.expectApproxEqAbs(@as(f64, 0.76519768655796655145), j0(1.0), eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0.22389077914123566805), j0(2.0), eps);
    try std.testing.expectApproxEqAbs(j0(1.5), j0(-1.5), eps);
}

test "j1 special and known values" {
    const eps = 1.0e-10;
    try std.testing.expectEqual(0.0, j1(0.0));
    try std.testing.expectEqual(0.0, j1(std.math.inf(f64)));
    try std.testing.expect(std.math.isNan(j1(std.math.nan(f64))));
    try std.testing.expectApproxEqAbs(@as(f64, 0.44005058574493351595), j1(1.0), eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0.57672480775687339920), j1(2.0), eps);
    try std.testing.expectApproxEqAbs(-j1(1.5), j1(-1.5), eps);
}

test "jn recurrence and known values" {
    const eps = 1.0e-9;
    const x = 1.5;
    try std.testing.expectEqual(j0(x), jn(0, x));
    try std.testing.expectEqual(j1(x), jn(1, x));
    try std.testing.expectApproxEqAbs((2.0 / x) * j1(x) - j0(x), jn(2, x), eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0.11490348493190048046), jn(2, 1.0), eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0.35283402861563772890), jn(2, 2.0), eps);
    try std.testing.expectEqual(0.0, jn(2, 0.0));
    try std.testing.expectEqual(0.0, jn(3, std.math.inf(f64)));
    try std.testing.expectApproxEqAbs(jn(4, 3.0), jn(-4, -3.0), eps);
}

test "y0 special and known values" {
    const eps = 1.0e-10;
    try std.testing.expect(std.math.isNegativeInf(y0(0.0)));
    try std.testing.expectEqual(0.0, y0(std.math.inf(f64)));
    try std.testing.expect(std.math.isNan(y0(-1.0)));
    try std.testing.expect(std.math.isNan(y0(std.math.nan(f64))));
    try std.testing.expectApproxEqAbs(@as(f64, 0.08825696421567695798), y0(1.0), eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0.51037567264974511960), y0(2.0), eps);
}

test "y1 special and known values" {
    const eps = 1.0e-10;
    try std.testing.expect(std.math.isNegativeInf(y1(0.0)));
    try std.testing.expectEqual(0.0, y1(std.math.inf(f64)));
    try std.testing.expect(std.math.isNan(y1(-1.0)));
    try std.testing.expect(std.math.isNan(y1(std.math.nan(f64))));
    try std.testing.expectApproxEqAbs(@as(f64, -0.78121282130028871654), y1(1.0), eps);
    try std.testing.expectApproxEqAbs(@as(f64, -0.10703243154093754689), y1(2.0), eps);
}

test "yn recurrence and known values" {
    const eps = 1.0e-9;
    const x = 1.5;
    try std.testing.expectEqual(y0(x), yn(0, x));
    try std.testing.expectEqual(y1(x), yn(1, x));
    try std.testing.expectApproxEqAbs((2.0 / x) * y1(x) - y0(x), yn(2, x), eps);
    try std.testing.expectApproxEqAbs(@as(f64, -1.65068260681625439393), yn(2, 1.0), eps);
    try std.testing.expectApproxEqAbs(@as(f64, -0.61740810419068266788), yn(2, 2.0), eps);
    try std.testing.expect(std.math.isNegativeInf(yn(2, 0.0)));
    try std.testing.expectEqual(0.0, yn(3, std.math.inf(f64)));
}
