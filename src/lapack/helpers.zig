const std = @import("std");
const float = @import("../float.zig");

/// Compute sqrt(x^2 + y^2) without overflow/underflow.
pub fn dlapy2(x: f64, y: f64) f64 {
    const ax = @abs(x);
    const ay = @abs(y);
    const a = @max(ax, ay);
    const b = @min(ax, ay);
    if (a == 0.0) return 0.0;
    const t = b / a;
    return a * @sqrt(1.0 + t * t);
}

/// Generate a real elementary Householder reflector H so that
/// H * x = (beta, 0, ..., 0)^T.
///
/// On entry, x[0] is the first component and x[1..n] (accessed with stride
/// incx) holds the tail. On exit, x[0] is overwritten with beta and the tail
/// holds the scaled reflector vector v[1..n] with v[0] = 1 implicit.
/// The scalar tau for H = I - tau * v * v^T is returned.
pub fn dlarfg(n: usize, x: []f64, incx: usize) f64 {
    std.debug.assert(incx >= 1);
    if (n <= 1) return 0.0;
    std.debug.assert(x.len >= 1 + (n - 1) * incx);

    const alpha = x[0];

    // Scale by the maximum absolute value to avoid overflow/underflow
    // while computing the norm.
    var scale: f64 = @abs(alpha);
    var i: usize = 1;
    while (i < n) : (i += 1) {
        scale = @max(scale, @abs(x[i * incx]));
    }

    if (scale == 0.0) {
        return 0.0;
    }

    const alpha_s = alpha / scale;

    var xnorm_sq: f64 = 0.0;
    i = 1;
    while (i < n) : (i += 1) {
        const t = x[i * incx] / scale;
        xnorm_sq += t * t;
    }
    const xnorm_s = @sqrt(xnorm_sq);

    // Use the raw sign bit so that -0.0 is treated as negative.
    const sign_alpha = if ((@as(u64, @bitCast(alpha)) >> 63) == 1) @as(f64, -1.0) else @as(f64, 1.0);
    const beta_s = -sign_alpha * dlapy2(alpha_s, xnorm_s);
    const beta = beta_s * scale;
    const tau = (beta_s - alpha_s) / beta_s;
    const divisor_s = alpha_s - beta_s;

    i = 1;
    while (i < n) : (i += 1) {
        x[i * incx] = (x[i * incx] / scale) / divisor_s;
    }
    x[0] = beta;

    return tau;
}

pub const Side = enum {
    left,
    right,
};

/// Apply an elementary reflector H = I - tau * v * v^T to a matrix C.
/// Only side == .left is implemented for this task.
///
/// C is stored in row-major order: c[i * ldc + j]. The vector v has length m
/// for side.left and is accessed with stride incv.
pub fn dlarf(side: Side, m: usize, n: usize, v: []const f64, incv: usize, tau: f64, c: []f64, ldc: usize) void {
    std.debug.assert(incv >= 1);
    if (m == 0 or n == 0 or tau == 0.0) return;

    switch (side) {
        .left => {
            std.debug.assert(v.len >= 1 + (m - 1) * incv);
            std.debug.assert(c.len >= (m - 1) * ldc + n);

            for (0..n) |j| {
                var dot: f64 = 0.0;
                for (0..m) |idx| {
                    dot += v[idx * incv] * c[idx * ldc + j];
                }
                const scale = -tau * dot;
                for (0..m) |idx| {
                    c[idx * ldc + j] += scale * v[idx * incv];
                }
            }
        },
        .right => {
            // Not required for this task.
            @panic("dlarf side.right is not implemented");
        },
    }
}

test "dlapy2 basic values" {
    try std.testing.expect(float.approxEqAbs(f64, dlapy2(3.0, 4.0), 5.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, dlapy2(5.0, 12.0), 13.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, dlapy2(0.0, 0.0), 0.0, 1e-12));
}

test "dlarfg annihilates tail" {
    var x = [_]f64{ 3.0, 4.0 };
    const tau = dlarfg(2, &x, 1);

    // After the call: x[0] = beta, x[1] = v[1] with v[0] = 1 implicit.
    try std.testing.expect(float.approxEqAbs(f64, x[0], -5.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, x[1], 0.5, 1e-12));

    // Verify H * [3, 4] = [beta, 0].
    const v = [_]f64{ 1.0, x[1] };
    const original = [_]f64{ 3.0, 4.0 };
    var result: [2]f64 = undefined;
    var dot: f64 = 0.0;
    for (0..2) |idx| dot += v[idx] * original[idx];
    for (0..2) |idx| result[idx] = original[idx] - tau * v[idx] * dot;

    try std.testing.expect(float.approxEqAbs(f64, result[0], x[0], 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, result[1], 0.0, 1e-12));
}

test "dlarf left applies reflector" {
    // Reflector for [3, 4]: v = [1, 0.5], tau = 1.6.
    var v = [_]f64{ 1.0, 0.5 };
    var c = [_]f64{
        3.0, 1.0,
        4.0, 2.0,
    };
    dlarf(.left, 2, 2, &v, 1, 1.6, &c, 2);

    // Row-major result of H * C.
    try std.testing.expect(float.approxEqAbs(f64, c[0], -5.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, c[1], -2.2, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, c[2], 0.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, c[3], 0.4, 1e-12));
}

test "dlarfg on unit vector" {
    var x = [_]f64{ 1.0, 1.0, 1.0 };
    const tau = dlarfg(3, &x, 1);
    const expected_beta = -std.math.sqrt(3.0);
    try std.testing.expect(float.approxEqAbs(f64, x[0], expected_beta, 1e-12));

    // Verify H * [1, 1, 1] = [beta, 0, 0].
    const original = [_]f64{ 1.0, 1.0, 1.0 };
    var dot: f64 = 0.0;
    for (0..3) |idx| {
        const vi = if (idx == 0) @as(f64, 1.0) else x[idx];
        dot += vi * original[idx];
    }
    for (0..3) |idx| {
        const vi = if (idx == 0) @as(f64, 1.0) else x[idx];
        const result = original[idx] - tau * vi * dot;
        if (idx == 0) {
            try std.testing.expect(float.approxEqAbs(f64, result, x[0], 1e-12));
        } else {
            try std.testing.expect(float.approxEqAbs(f64, result, 0.0, 1e-12));
        }
    }
}
