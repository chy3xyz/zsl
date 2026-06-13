const std = @import("std");

// Coefficients for the gamma rational approximation on [1, 2].
const gamma_p = [_]f64{
    1.60119522476751861407e-04,
    1.19135147006586384913e-03,
    1.04213797561761569935e-02,
    4.76367800457137231464e-02,
    2.07448227648435975150e-01,
    4.94214826801497100753e-01,
    9.99999999999999996796e-01,
};

const gamma_q = [_]f64{
    -2.31581873324120129819e-05,
    5.39605580493303397842e-04,
    -4.45641913851797240494e-03,
    1.18139785222060435552e-02,
    3.58236398605498653373e-02,
    -2.34591795718243348568e-01,
    7.14304917030273074085e-02,
    1.00000000000000000320e+00,
};

// Coefficients for Stirling's approximation.
const gamma_s = [_]f64{
    7.87311395793093628397e-04,
    -2.29549961613378126380e-04,
    -2.68132617805781232825e-03,
    3.47222221605458667310e-03,
    8.33333333333482257126e-02,
};

// Coefficients for log_gamma.
const lgamma_a = [_]f64{
    7.72156649015328655494e-02,
    3.22467033424113591611e-01,
    6.73523010531292681824e-02,
    2.05808084325167332806e-02,
    7.38555086081402883957e-03,
    2.89051383673415629091e-03,
    1.19270763183362067845e-03,
    5.10069792153511336608e-04,
    2.20862790713908385557e-04,
    1.08011567247583939954e-04,
    2.52144565451257326939e-05,
    4.48640949618915160150e-05,
};

const lgamma_r = [_]f64{
    1.0,
    1.39200533467621045958e+00,
    7.21935547567138069525e-01,
    1.71933865632803078993e-01,
    1.86459191715652901344e-02,
    7.77942496381893596434e-04,
    7.32668430744625636189e-06,
};

const lgamma_s = [_]f64{
    -7.72156649015328655494e-02,
    2.14982415960608852501e-01,
    3.25778796408930981787e-01,
    1.46350472652464452805e-01,
    2.66422703033638609560e-02,
    1.84028451407337715652e-03,
    3.19475326584100867617e-05,
};

const lgamma_t = [_]f64{
    4.83836122723810047042e-01,
    -1.47587722994593911752e-01,
    6.46249402391333854778e-02,
    -3.27885410759859649565e-02,
    1.79706750811820387126e-02,
    -1.03142241298341437450e-02,
    6.10053870246291332635e-03,
    -3.68452016781138256760e-03,
    2.25964780900612472250e-03,
    -1.40346469989232843813e-03,
    8.81081882437654011382e-04,
    -5.38595305356740546715e-04,
    3.15632070903625950361e-04,
    -3.12754168375120860518e-04,
    3.35529192635519073543e-04,
};

const lgamma_u = [_]f64{
    -7.72156649015328655494e-02,
    6.32827064025093366517e-01,
    1.45492250137234768737e+00,
    9.77717527963372745603e-01,
    2.28963728064692451092e-01,
    1.33810918536787660377e-02,
};

const lgamma_v = [_]f64{
    1.0,
    2.45597793713041134822e+00,
    2.12848976379893395361e+00,
    7.69285150456672783825e-01,
    1.04222645593369134254e-01,
    3.21709242282423911810e-03,
};

const lgamma_w = [_]f64{
    4.18938533204672725052e-01,
    8.33333333333329678849e-02,
    -2.77777777728775536470e-03,
    7.93650558643019558500e-04,
    -5.95187557450339963135e-04,
    8.36339918996282139126e-04,
    -1.63092934096575273989e-03,
};

fn is_neg_int(x: f64) bool {
    if (x >= 0.0) return false;
    const r = std.math.modf(x);
    return r.fpart == 0.0;
}

// Stirling's formula for the gamma function.
// The caller must multiply the two returned values together.
fn stirling(x: f64) struct { f64, f64 } {
    if (x > 200.0) {
        return .{ std.math.inf(f64), 1.0 };
    }
    const sqrt_two_pi = 2.506628274631000502417;
    const max_stirling = 143.01608;
    var w = 1.0 / x;
    w = 1.0 +
        w * ((((gamma_s[0] * w + gamma_s[1]) * w + gamma_s[2]) * w + gamma_s[3]) * w + gamma_s[4]);
    var y1 = @exp(x);
    var y2: f64 = 1.0;
    if (x > max_stirling) {
        const v = std.math.pow(f64, x, 0.5 * x - 0.25);
        const y1_ = y1;
        y1 = v;
        y2 = v / y1_;
    } else {
        y1 = std.math.pow(f64, x, x - 0.5) / y1;
    }
    return .{ y1, sqrt_two_pi * w * y2 };
}

/// gamma returns the gamma function of x.
///
/// Special cases:
/// - gamma(+inf) = +inf
/// - gamma(+0)   = +inf
/// - gamma(-0)   = -inf
/// - gamma(x)    = nan for negative integers
/// - gamma(-inf) = nan
/// - gamma(nan)  = nan
pub fn gamma(x_: f64) f64 {
    const euler = 0.57721566490153286060651209008240243104215933593992;
    var x = x_;

    if (is_neg_int(x) or std.math.isNegativeInf(x) or std.math.isNan(x)) {
        return std.math.nan(f64);
    }
    if (std.math.isPositiveInf(x)) {
        return std.math.inf(f64);
    }
    if (x == 0.0) {
        return std.math.copysign(std.math.inf(f64), x);
    }

    const q = @abs(x);
    var p = @floor(q);

    if (q > 33.0) {
        if (x >= 0.0) {
            const y = stirling(x);
            return y[0] * y[1];
        }

        var signgam: i32 = 1;
        if (p <= 9.22e18) {
            const ip: i64 = @intFromFloat(p);
            if ((ip & 1) == 0) {
                signgam = -1;
            }
        } else {
            signgam = -1;
        }

        var z = q - p;
        if (z > 0.5) {
            p += 1.0;
            z = q - p;
        }
        z = q * @sin(std.math.pi * z);
        if (z == 0.0) {
            return std.math.copysign(std.math.inf(f64), @as(f64, @floatFromInt(signgam)));
        }
        const s = stirling(q);
        const absz = @abs(z);
        const d = absz * s[0] * s[1];
        if (std.math.isInf(d)) {
            z = std.math.pi / absz / s[0] / s[1];
        } else {
            z = std.math.pi / d;
        }
        return @as(f64, @floatFromInt(signgam)) * z;
    }

    // Reduce argument.
    var z: f64 = 1.0;

    while (x >= 3.0) {
        x -= 1.0;
        z *= x;
    }

    while (x < 0.0) {
        if (x > -1.0e-9) {
            if (x == 0.0) return std.math.inf(f64);
            return z / ((1.0 + euler * x) * x);
        }
        z /= x;
        x += 1.0;
    }

    while (x < 2.0) {
        if (x < 1.0e-9) {
            if (x == 0.0) return std.math.inf(f64);
            return z / ((1.0 + euler * x) * x);
        }
        z /= x;
        x += 1.0;
    }

    if (x == 2.0) {
        return z;
    }

    x -= 2.0;
    const p_poly = (((((x * gamma_p[0] + gamma_p[1]) * x + gamma_p[2]) * x + gamma_p[3]) * x +
        gamma_p[4]) * x + gamma_p[5]) * x + gamma_p[6];
    const q_poly = ((((((x * gamma_q[0] + gamma_q[1]) * x + gamma_q[2]) * x + gamma_q[3]) * x +
        gamma_q[4]) * x + gamma_q[5]) * x + gamma_q[6]) * x + gamma_q[7];
    return z * p_poly / q_poly;
}

/// Helper: returns log(|Gamma(x)|) and the sign of Gamma(x).
fn log_gamma_sign(x_: f64) struct { f64, i32 } {
    var x = x_;
    const ymin = 1.461632144968362245;
    const tiny = @exp2(-70.0);
    const two52 = @exp2(52.0);
    const two58 = @exp2(58.0);
    const tc = 1.46163214496836224576e+00;
    const tf = -1.21486290535849611461e-01;
    const tt = -3.63867699703950536541e-18;

    var sign: i32 = 1;

    if (std.math.isNan(x)) {
        return .{ x, sign };
    }
    if (std.math.isPositiveInf(x)) {
        return .{ x, sign };
    }
    if (x == 0.0) {
        return .{ std.math.inf(f64), sign };
    }

    var neg = false;
    if (x < 0.0) {
        x = -x;
        neg = true;
    }

    if (x < tiny) {
        if (neg) {
            sign = -1;
        }
        return .{ -@log(x), sign };
    }

    var nadj: f64 = 0.0;
    if (neg) {
        if (x >= two52) {
            return .{ std.math.inf(f64), sign };
        }
        const t = sin_pi(x);
        if (t == 0.0) {
            return .{ std.math.inf(f64), sign };
        }
        nadj = @log(std.math.pi / @abs(t * x));
        if (t < 0.0) {
            sign = -1;
        }
    }

    var lgamma: f64 = 0.0;
    if (x == 1.0 or x == 2.0) {
        return .{ 0.0, sign };
    } else if (x < 2.0) {
        var y: f64 = 0.0;
        var i: usize = 0;
        if (x <= 0.9) {
            lgamma = -@log(x);
            if (x >= (ymin - 1.0 + 0.27)) {
                y = 1.0 - x;
                i = 0;
            } else if (x >= (ymin - 1.0 - 0.27)) {
                y = x - (tc - 1.0);
                i = 1;
            } else {
                y = x;
                i = 2;
            }
        } else {
            lgamma = 0.0;
            if (x >= (ymin + 0.27)) {
                y = 2.0 - x;
                i = 0;
            } else if (x >= (ymin - 0.27)) {
                y = x - tc;
                i = 1;
            } else {
                y = x - 1.0;
                i = 2;
            }
        }

        if (i == 0) {
            const z2 = y * y;
            const gamma_p1 = lgamma_a[0] +
                z2 * (lgamma_a[2] +
                    z2 * (lgamma_a[4] +
                        z2 * (lgamma_a[6] +
                            z2 * (lgamma_a[8] + z2 * lgamma_a[10]))));
            const gamma_p2 = z2 * (lgamma_a[1] +
                z2 * (lgamma_a[3] +
                    z2 * (lgamma_a[5] +
                        z2 * (lgamma_a[7] +
                            z2 * (lgamma_a[9] + z2 * lgamma_a[11])))));
            const p = y * gamma_p1 + gamma_p2;
            lgamma += (p - 0.5 * y);
        } else if (i == 1) {
            const z2 = y * y;
            const w = z2 * y;
            const gamma_p1 = lgamma_t[0] +
                w * (lgamma_t[3] + w * (lgamma_t[6] + w * (lgamma_t[9] + w * lgamma_t[12])));
            const gamma_p2 = lgamma_t[1] +
                w * (lgamma_t[4] + w * (lgamma_t[7] + w * (lgamma_t[10] + w * lgamma_t[13])));
            const gamma_p3 = lgamma_t[2] +
                w * (lgamma_t[5] + w * (lgamma_t[8] + w * (lgamma_t[11] + w * lgamma_t[14])));
            const p = z2 * gamma_p1 - (tt - w * (gamma_p2 + y * gamma_p3));
            lgamma += (tf + p);
        } else {
            const gamma_p1 = y * (lgamma_u[0] +
                y * (lgamma_u[1] +
                    y * (lgamma_u[2] +
                        y * (lgamma_u[3] +
                            y * (lgamma_u[4] + y * lgamma_u[5])))));
            const gamma_p2 = 1.0 +
                y * (lgamma_v[1] +
                    y * (lgamma_v[2] +
                        y * (lgamma_v[3] +
                            y * (lgamma_v[4] + y * lgamma_v[5]))));
            lgamma += (-0.5 * y + gamma_p1 / gamma_p2);
        }
    } else if (x < 8.0) {
        const i = @as(usize, @intFromFloat(x));
        const y = x - @as(f64, @floatFromInt(i));
        const p = y * (lgamma_s[0] +
            y * (lgamma_s[1] +
                y * (lgamma_s[2] +
                    y * (lgamma_s[3] +
                        y * (lgamma_s[4] +
                            y * (lgamma_s[5] + y * lgamma_s[6]))))));
        const q = 1.0 +
            y * (lgamma_r[1] +
                y * (lgamma_r[2] +
                    y * (lgamma_r[3] +
                        y * (lgamma_r[4] +
                            y * (lgamma_r[5] + y * lgamma_r[6])))));
        lgamma = 0.5 * y + p / q;
        var z_prod: f64 = 1.0;
        if (i == 7) {
            z_prod *= (y + 6.0);
            z_prod *= (y + 5.0);
            z_prod *= (y + 4.0);
            z_prod *= (y + 3.0);
            z_prod *= (y + 2.0);
            lgamma += @log(z_prod);
        } else if (i == 6) {
            z_prod *= (y + 5.0);
            z_prod *= (y + 4.0);
            z_prod *= (y + 3.0);
            z_prod *= (y + 2.0);
            lgamma += @log(z_prod);
        } else if (i == 5) {
            z_prod *= (y + 4.0);
            z_prod *= (y + 3.0);
            z_prod *= (y + 2.0);
            lgamma += @log(z_prod);
        } else if (i == 4) {
            z_prod *= (y + 3.0);
            z_prod *= (y + 2.0);
            lgamma += @log(z_prod);
        } else if (i == 3) {
            z_prod *= (y + 2.0);
            lgamma += @log(z_prod);
        }
    } else if (x < two58) {
        const t = @log(x);
        const z2 = 1.0 / x;
        const y2 = z2 * z2;
        const w = lgamma_w[0] +
            z2 * (lgamma_w[1] +
                y2 * (lgamma_w[2] +
                    y2 * (lgamma_w[3] +
                        y2 * (lgamma_w[4] +
                            y2 * (lgamma_w[5] + y2 * lgamma_w[6])))));
        lgamma = (x - 0.5) * (t - 1.0) + w;
    } else {
        lgamma = x * (@log(x) - 1.0);
    }

    if (neg) {
        lgamma = nadj - lgamma;
    }
    return .{ lgamma, sign };
}

/// sin_pi(x) is a helper for negative log_gamma arguments.
fn sin_pi(x_: f64) f64 {
    var x = x_;
    const two52 = @exp2(52.0);
    const two53 = @exp2(53.0);
    if (x < 0.25) {
        return -@sin(std.math.pi * x);
    }

    var z = @floor(x);
    var n: usize = 0;
    if (z != x) {
        x = @rem(x, 2.0);
        n = @as(usize, @intFromFloat(x * 4.0));
    } else {
        if (x >= two53) {
            x = 0.0;
            n = 0;
        } else {
            if (x < two52) {
                z = x + two52;
            }
            n = @as(usize, @intCast(@as(u64, @bitCast(z)) & 1));
            x = @as(f64, @floatFromInt(n));
            n <<= 2;
        }
    }

    if (n == 0) {
        x = @sin(std.math.pi * x);
    } else if (n == 1 or n == 2) {
        x = @cos(std.math.pi * (0.5 - x));
    } else if (n == 3 or n == 4) {
        x = @sin(std.math.pi * (1.0 - x));
    } else if (n == 5 or n == 6) {
        x = -@cos(std.math.pi * (x - 1.5));
    } else {
        x = @sin(std.math.pi * (x - 2.0));
    }
    return -x;
}

/// log_gamma returns the natural logarithm of the absolute value of Gamma(x).
///
/// Special cases:
/// - log_gamma(+inf)  = +inf
/// - log_gamma(0)     = +inf
/// - log_gamma(-integer) = +inf
/// - log_gamma(-inf)  = -inf
/// - log_gamma(nan)   = nan
pub fn log_gamma(x: f64) f64 {
    const r = log_gamma_sign(x);
    return r[0];
}

/// factorial returns n! as a floating-point number.
pub fn factorial(n: usize) f64 {
    return gamma(@as(f64, @floatFromInt(n)) + 1.0);
}

/// ln_factorial returns the natural logarithm of n!.
pub fn ln_factorial(n: usize) f64 {
    return log_gamma(@as(f64, @floatFromInt(n)) + 1.0);
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

test "gamma reference values" {
    const eps = 1e-12;
    try std.testing.expectApproxEqAbs(gamma(5.0), 24.0, eps);
    try std.testing.expectApproxEqAbs(gamma(0.5), std.math.sqrt(std.math.pi), eps);
    try std.testing.expect(std.math.isPositiveInf(gamma(0.0)));
    try std.testing.expect(std.math.isNegativeInf(gamma(-0.0)));
    try std.testing.expect(std.math.isNan(gamma(-1.0)));
}

test "log_gamma reference values" {
    const eps = 1e-12;
    try std.testing.expectApproxEqAbs(log_gamma(1.0), 0.0, eps);
    try std.testing.expectApproxEqAbs(log_gamma(2.0), 0.0, eps);
    try std.testing.expectApproxEqAbs(log_gamma(5.0), @log(24.0), eps);
    try std.testing.expectApproxEqAbs(log_gamma(0.5), 0.5 * @log(std.math.pi), eps);
}

test "factorial and ln_factorial" {
    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(factorial(0), 1.0, eps);
    try std.testing.expectApproxEqAbs(factorial(5), 120.0, eps);
    try std.testing.expectApproxEqAbs(factorial(10), 3628800.0, eps);
    try std.testing.expectApproxEqAbs(ln_factorial(0), 0.0, eps);
    try std.testing.expectApproxEqAbs(ln_factorial(5), @log(120.0), eps);
    try std.testing.expectApproxEqAbs(ln_factorial(10), @log(3628800.0), eps);
}
