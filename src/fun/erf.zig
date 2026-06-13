const std = @import("std");

const erx = 8.45062911510467529297e-01;

// Coefficients for approximation to erf in [0, 0.84375].
const efx = 1.28379167095512586316e-01;
const efx8 = 1.02703333676410069053e+00;

const pp0 = 1.28379167095512558561e-01;
const pp1 = -3.25042107247001499370e-01;
const pp2 = -2.84817495755985104766e-02;
const pp3 = -5.77027029648944159157e-03;
const pp4 = -2.37630166566501626084e-05;

const qq1 = 3.97917223959155352819e-01;
const qq2 = 6.50222499887672944485e-02;
const qq3 = 5.08130628187576562776e-03;
const qq4 = 1.32494738004321644526e-04;
const qq5 = -3.96022827877536812320e-06;

// Coefficients for approximation to erf in [0.84375, 1.25].
const pa0 = -2.36211856075265944077e-03;
const pa1 = 4.14856118683748331666e-01;
const pa2 = -3.72207876035701323847e-01;
const pa3 = 3.18346619901161753674e-01;
const pa4 = -1.10894694282396677476e-01;
const pa5 = 3.54783043256182359371e-02;
const pa6 = -2.16637559486879084300e-03;

const qa1 = 1.06420880400844228286e-01;
const qa2 = 5.40397917702171048937e-01;
const qa3 = 7.18286544141962662868e-02;
const qa4 = 1.26171219808761642112e-01;
const qa5 = 1.36370839120290507362e-02;
const qa6 = 1.19844998467991074170e-02;

// Coefficients for approximation to erfc in [1.25, 1/0.35].
const ra0 = -9.86494403484714822705e-03;
const ra1 = -6.93858572707181764372e-01;
const ra2 = -1.05586262253232909814e+01;
const ra3 = -6.23753324503260060396e+01;
const ra4 = -1.62396669462573470355e+02;
const ra5 = -1.84605092906711035994e+02;
const ra6 = -8.12874355063065934246e+01;
const ra7 = -9.81432934416914548592e+00;

const sa1 = 1.96512716674392571292e+01;
const sa2 = 1.37657754143519042600e+02;
const sa3 = 4.34565877475229228821e+02;
const sa4 = 6.45387271733267880336e+02;
const sa5 = 4.29008140027567833386e+02;
const sa6 = 1.08635005541779435134e+02;
const sa7 = 6.57024977031928170135e+00;
const sa8 = -6.04244152148580987438e-02;

// Coefficients for approximation to erfc in [1/.35, 28].
const rb0 = -9.86494292470009928597e-03;
const rb1 = -7.99283237680523006574e-01;
const rb2 = -1.77579549177547519889e+01;
const rb3 = -1.60636384855821916062e+02;
const rb4 = -6.37566443368389627722e+02;
const rb5 = -1.02509513161107724954e+03;
const rb6 = -4.83519191608651397019e+02;

const sb1 = 3.03380607434824582924e+01;
const sb2 = 3.25792512996573918826e+02;
const sb3 = 1.53672958608443695994e+03;
const sb4 = 3.19985821950859553908e+03;
const sb5 = 2.55305040643316442583e+03;
const sb6 = 4.74528541206955367215e+02;
const sb7 = -2.24409524465858183362e+01;

/// erf returns the error function of x.
///
/// Special cases:
/// - erf(+inf) = 1
/// - erf(-inf) = -1
/// - erf(nan)  = nan
pub fn erf(x_: f64) f64 {
    var x = x_;
    const very_tiny = 2.848094538889218e-306;
    const small = 1.0 / @as(f64, 1 << 28);

    if (std.math.isNan(x)) {
        return std.math.nan(f64);
    }
    if (std.math.isPositiveInf(x)) {
        return 1.0;
    }
    if (std.math.isNegativeInf(x)) {
        return -1.0;
    }

    var sign = false;
    if (x < 0.0) {
        x = -x;
        sign = true;
    }

    if (x < 0.84375) {
        var temp: f64 = 0.0;
        if (x < small) {
            if (x < very_tiny) {
                temp = 0.125 * (8.0 * x + efx8 * x);
            } else {
                temp = x + efx * x;
            }
        } else {
            const z = x * x;
            const r = pp0 + z * (pp1 + z * (pp2 + z * (pp3 + z * pp4)));
            const s = 1.0 + z * (qq1 + z * (qq2 + z * (qq3 + z * (qq4 + z * qq5))));
            const y = r / s;
            temp = x + x * y;
        }
        return if (sign) -temp else temp;
    }

    if (x < 1.25) {
        const s = x - 1.0;
        const p = pa0 + s * (pa1 + s * (pa2 + s * (pa3 + s * (pa4 + s * (pa5 + s * pa6)))));
        const q = 1.0 + s * (qa1 + s * (qa2 + s * (qa3 + s * (qa4 + s * (qa5 + s * qa6)))));
        if (sign) {
            return -erx - p / q;
        }
        return erx + p / q;
    }

    if (x >= 6.0) {
        return if (sign) -1.0 else 1.0;
    }

    const s = 1.0 / (x * x);
    var r: f64 = 0.0;
    var t: f64 = 0.0;
    if (x < 1.0 / 0.35) {
        r = ra0 +
            s * (ra1 + s * (ra2 + s * (ra3 + s * (ra4 + s * (ra5 + s * (ra6 + s * ra7))))));
        t = 1.0 +
            s * (sa1 + s * (sa2 + s * (sa3 + s * (sa4 + s * (sa5 + s * (sa6 + s * (sa7 + s * sa8)))))));
    } else {
        r = rb0 + s * (rb1 + s * (rb2 + s * (rb3 + s * (rb4 + s * (rb5 + s * rb6)))));
        t = 1.0 +
            s * (sb1 + s * (sb2 + s * (sb3 + s * (sb4 + s * (sb5 + s * (sb6 + s * sb7))))));
    }

    const z = @as(f64, @bitCast(@as(u64, @bitCast(x)) & 0xffffffff00000000));
    const r_ = @exp(-z * z - 0.5625) * @exp((z - x) * (z + x) + r / t);
    if (sign) {
        return r_ / x - 1.0;
    }
    return 1.0 - r_ / x;
}

/// erfc returns the complementary error function of x.
///
/// Special cases:
/// - erfc(+inf) = 0
/// - erfc(-inf) = 2
/// - erfc(nan)  = nan
pub fn erfc(x_: f64) f64 {
    var x = x_;
    const tiny = 1.0 / @as(f64, 1 << 56);

    if (std.math.isNan(x)) {
        return std.math.nan(f64);
    }
    if (std.math.isPositiveInf(x)) {
        return 0.0;
    }
    if (std.math.isNegativeInf(x)) {
        return 2.0;
    }

    var sign = false;
    if (x < 0.0) {
        x = -x;
        sign = true;
    }

    if (x < 0.84375) {
        var temp: f64 = 0.0;
        if (x < tiny) {
            temp = x;
        } else {
            const z = x * x;
            const r = pp0 + z * (pp1 + z * (pp2 + z * (pp3 + z * pp4)));
            const s = 1.0 + z * (qq1 + z * (qq2 + z * (qq3 + z * (qq4 + z * qq5))));
            const y = r / s;
            if (x < 0.25) {
                temp = x + x * y;
            } else {
                temp = 0.5 + (x * y + (x - 0.5));
            }
        }
        if (sign) {
            return 1.0 + temp;
        }
        return 1.0 - temp;
    }

    if (x < 1.25) {
        const s = x - 1.0;
        const p = pa0 + s * (pa1 + s * (pa2 + s * (pa3 + s * (pa4 + s * (pa5 + s * pa6)))));
        const q = 1.0 + s * (qa1 + s * (qa2 + s * (qa3 + s * (qa4 + s * (qa5 + s * qa6)))));
        if (sign) {
            return 1.0 + erx + p / q;
        }
        return 1.0 - erx - p / q;
    }

    if (x < 28.0) {
        const s = 1.0 / (x * x);
        var r: f64 = 0.0;
        var t: f64 = 0.0;
        if (x < 1.0 / 0.35) {
            r = ra0 +
                s * (ra1 + s * (ra2 + s * (ra3 + s * (ra4 + s * (ra5 + s * (ra6 + s * ra7))))));
            t = 1.0 +
                s * (sa1 + s * (sa2 + s * (sa3 + s * (sa4 + s * (sa5 + s * (sa6 + s * (sa7 + s * sa8)))))));
        } else {
            if (sign and x > 6.0) {
                return 2.0;
            }
            r = rb0 + s * (rb1 + s * (rb2 + s * (rb3 + s * (rb4 + s * (rb5 + s * rb6)))));
            t = 1.0 +
                s * (sb1 + s * (sb2 + s * (sb3 + s * (sb4 + s * (sb5 + s * (sb6 + s * sb7))))));
        }

        const z = @as(f64, @bitCast(@as(u64, @bitCast(x)) & 0xffffffff00000000));
        const r_ = @exp(-z * z - 0.5625) * @exp((z - x) * (z + x) + r / t);
        if (sign) {
            return 2.0 - r_ / x;
        }
        return r_ / x;
    }

    if (sign) {
        return 2.0;
    }
    return 0.0;
}

/// erfcx returns the scaled complementary error function,
///   erfcx(x) = exp(x^2) * erfc(x).
///
/// Special cases:
/// - erfcx(+inf) = 0
/// - erfcx(-inf) = +inf
/// - erfcx(nan)  = nan
pub fn erfcx(x: f64) f64 {
    if (std.math.isNan(x)) {
        return std.math.nan(f64);
    }
    if (std.math.isPositiveInf(x)) {
        return 0.0;
    }
    if (std.math.isNegativeInf(x)) {
        return std.math.inf(f64);
    }

    const x2 = x * x;
    // Avoid overflow of exp(x^2); for large positive x use the asymptotic form.
    if (x2 > 700.0) {
        if (x > 0.0) {
            return 1.0 / (std.math.sqrt(std.math.pi) * x);
        }
        return std.math.inf(f64);
    }

    if (x >= 0.0) {
        return @exp(x2) * erfc(x);
    }
    return 2.0 * @exp(x2) - erfcx(-x);
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

test "erf reference values" {
    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(erf(0.0), 0.0, eps);
    try std.testing.expectApproxEqAbs(erf(1.0), 0.8427007929, eps);
    try std.testing.expectApproxEqAbs(erf(-1.0), -0.8427007929, eps);
    try std.testing.expect(erf(std.math.inf(f64)) == 1.0);
    try std.testing.expect(erf(-std.math.inf(f64)) == -1.0);
    try std.testing.expect(std.math.isNan(erf(std.math.nan(f64))));
}

test "erfc reference values" {
    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(erfc(0.0), 1.0, eps);
    try std.testing.expectApproxEqAbs(erfc(1.0), 1.0 - 0.8427007929, eps);
    try std.testing.expectApproxEqAbs(erfc(-1.0), 1.0 + 0.8427007929, eps);
    try std.testing.expectApproxEqAbs(erfc(std.math.inf(f64)), 0.0, eps);
    try std.testing.expectApproxEqAbs(erfc(-std.math.inf(f64)), 2.0, eps);
    try std.testing.expect(std.math.isNan(erfc(std.math.nan(f64))));
}

test "erfcx reference values" {
    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(erfcx(0.0), 1.0, eps);
    try std.testing.expectApproxEqAbs(erfcx(1.0), @exp(1.0) * erfc(1.0), eps);
    try std.testing.expectApproxEqAbs(erfcx(-1.0), @exp(1.0) * erfc(-1.0), eps);
    try std.testing.expect(std.math.isPositiveInf(erfcx(-std.math.inf(f64))));
    try std.testing.expect(erfcx(std.math.inf(f64)) == 0.0);
}
