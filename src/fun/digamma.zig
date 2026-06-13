const std = @import("std");

const eul = 0.57721566490153286061;

// Coefficients for the asymptotic expansion of digamma.
const digamma_a = [_]f64{
    8.33333333333333333333e-2,
    -2.10927960927960927961e-2,
    7.57575757575757575758e-3,
    -4.16666666666666666667e-3,
    3.96825396825396825397e-3,
    -8.33333333333333333333e-3,
    8.33333333333333333333e-2,
};

/// Evaluate the digamma asymptotic polynomial using Horner's method.
fn eval_digamma_poly(z: f64) f64 {
    var y: f64 = digamma_a[digamma_a.len - 1];
    var i: usize = digamma_a.len - 1;
    while (i > 0) {
        i -= 1;
        y = digamma_a[i] + z * y;
    }
    return y;
}

/// digamma returns the digamma function (the logarithmic derivative of the
/// gamma function) of x.
pub fn digamma(x_: f64) f64 {
    var x = x_;
    var negative = false;
    var nz: f64 = 0.0;
    var y: f64 = 0.0;

    if (x <= 0.0) {
        negative = true;
        const q = x;
        var p = @floor(q);
        if (p == q) {
            return std.math.inf(f64);
        }

        // Remove the zeros of tan(PI x) by subtracting the nearest integer.
        nz = q - p;
        if (nz != 0.5) {
            if (nz > 0.5) {
                p += 1.0;
                nz = q - p;
            }
            nz = std.math.pi / @tan(std.math.pi * nz);
        } else {
            nz = 0.0;
        }

        x = 1.0 - x;
    }

    // Check for positive integers up to 10.
    if (x <= 10.0 and x == @floor(x)) {
        const n: usize = @intFromFloat(x);
        for (1..n) |i| {
            y += 1.0 / @as(f64, @floatFromInt(i));
        }
        y -= eul;
    } else {
        var s = x;
        var w: f64 = 0.0;
        while (s < 10.0) {
            w += 1.0 / s;
            s += 1.0;
        }

        if (s < 1.0e17) {
            const z = 1.0 / (s * s);
            y = z * eval_digamma_poly(z);
        } else {
            y = 0.0;
        }

        y = @log(s) - (0.5 / s) - y - w;
    }

    if (negative) {
        y -= nz;
    }
    return y;
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

test "digamma reference values" {
    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(digamma(1.0), -0.5772156649, eps);
    try std.testing.expectApproxEqAbs(digamma(2.0), 1.0 - eul, eps);
    try std.testing.expect(std.math.isPositiveInf(digamma(0.0)));
}
