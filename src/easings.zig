const std = @import("std");
const math = std.math;

/// Default back overshoot. Using `1.0` matches the implicit coefficient in VSL's
/// `back_ease_*` implementations.
pub const default_back_overshoot: f64 = 1.0;

/// Default elastic overshoot. Using `1.0` matches the implicit amplitude in VSL's
/// `elastic_ease_*` implementations.
pub const default_elastic_overshoot: f64 = 1.0;

pub fn linear(t: f64) f64 {
    return t;
}

pub fn ease_in_quad(t: f64) f64 {
    return t * t;
}

pub fn ease_out_quad(t: f64) f64 {
    const f = 1.0 - t;
    return 1.0 - f * f;
}

pub fn ease_in_out_quad(t: f64) f64 {
    if (t < 0.5) {
        return 2.0 * t * t;
    } else {
        const f = -2.0 * t + 2.0;
        return 1.0 - f * f / 2.0;
    }
}

pub fn ease_in_cubic(t: f64) f64 {
    return t * t * t;
}

pub fn ease_out_cubic(t: f64) f64 {
    const f = 1.0 - t;
    return 1.0 - f * f * f;
}

pub fn ease_in_out_cubic(t: f64) f64 {
    if (t < 0.5) {
        return 4.0 * t * t * t;
    } else {
        const f = -2.0 * t + 2.0;
        return 1.0 - f * f * f / 2.0;
    }
}

pub fn ease_in_quart(t: f64) f64 {
    return t * t * t * t;
}

pub fn ease_out_quart(t: f64) f64 {
    const f = 1.0 - t;
    return 1.0 - f * f * f * f;
}

pub fn ease_in_out_quart(t: f64) f64 {
    if (t < 0.5) {
        return 8.0 * t * t * t * t;
    } else {
        const f = -2.0 * t + 2.0;
        return 1.0 - f * f * f * f / 2.0;
    }
}

pub fn ease_in_quint(t: f64) f64 {
    return t * t * t * t * t;
}

pub fn ease_out_quint(t: f64) f64 {
    const f = 1.0 - t;
    return 1.0 - f * f * f * f * f;
}

pub fn ease_in_out_quint(t: f64) f64 {
    if (t < 0.5) {
        return 16.0 * t * t * t * t * t;
    } else {
        const f = -2.0 * t + 2.0;
        return 1.0 - f * f * f * f * f / 2.0;
    }
}

pub fn ease_in_sine(t: f64) f64 {
    return 1.0 - math.cos(t * math.pi / 2.0);
}

pub fn ease_out_sine(t: f64) f64 {
    return math.sin(t * math.pi / 2.0);
}

pub fn ease_in_out_sine(t: f64) f64 {
    return 0.5 * (1.0 - math.cos(t * math.pi));
}

pub fn ease_in_circ(t: f64) f64 {
    return 1.0 - math.sqrt(1.0 - t * t);
}

pub fn ease_out_circ(t: f64) f64 {
    const f = t - 1.0;
    return math.sqrt(1.0 - f * f);
}

pub fn ease_in_out_circ(t: f64) f64 {
    if (t < 0.5) {
        return 0.5 * (1.0 - math.sqrt(1.0 - 4.0 * t * t));
    } else {
        const f = -2.0 * t + 2.0;
        return 0.5 * (math.sqrt(1.0 - f * f) + 1.0);
    }
}

pub fn ease_in_back(t: f64) f64 {
    return ease_in_back_overshoot(t, default_back_overshoot);
}

pub fn ease_in_back_overshoot(t: f64, overshoot: f64) f64 {
    return t * t * t - overshoot * t * math.sin(t * math.pi);
}

pub fn ease_out_back(t: f64) f64 {
    return ease_out_back_overshoot(t, default_back_overshoot);
}

pub fn ease_out_back_overshoot(t: f64, overshoot: f64) f64 {
    const f = 1.0 - t;
    return 1.0 - (f * f * f - overshoot * f * math.sin(f * math.pi));
}

pub fn ease_in_out_back(t: f64) f64 {
    return ease_in_out_back_overshoot(t, default_back_overshoot);
}

pub fn ease_in_out_back_overshoot(t: f64, overshoot: f64) f64 {
    if (t < 0.5) {
        const f = 2.0 * t;
        return 0.5 * (f * f * f - overshoot * f * math.sin(f * math.pi));
    } else {
        const f = 1.0 - (2.0 * t - 1.0);
        return 0.5 * (1.0 - (f * f * f - overshoot * f * math.sin(f * math.pi))) + 0.5;
    }
}

pub fn ease_in_elastic(t: f64) f64 {
    return ease_in_elastic_overshoot(t, default_elastic_overshoot);
}

pub fn ease_in_elastic_overshoot(t: f64, overshoot: f64) f64 {
    if (t == 0.0) return 0.0;
    if (t == 1.0) return 1.0;
    const period = 0.3;
    const amp = @max(overshoot, 1.0);
    const s = period / (2.0 * math.pi) * math.asin(1.0 / amp);
    return -(amp * math.pow(f64, 2.0, 10.0 * (t - 1.0)) * math.sin((t - 1.0 - s) * (2.0 * math.pi / period)));
}

pub fn ease_out_elastic(t: f64) f64 {
    return ease_out_elastic_overshoot(t, default_elastic_overshoot);
}

pub fn ease_out_elastic_overshoot(t: f64, overshoot: f64) f64 {
    if (t == 0.0) return 0.0;
    if (t == 1.0) return 1.0;
    const period = 0.3;
    const amp = @max(overshoot, 1.0);
    const s = period / (2.0 * math.pi) * math.asin(1.0 / amp);
    return amp * math.pow(f64, 2.0, -10.0 * t) * math.sin((t - s) * (2.0 * math.pi / period)) + 1.0;
}

pub fn ease_in_out_elastic(t: f64) f64 {
    return ease_in_out_elastic_overshoot(t, default_elastic_overshoot);
}

pub fn ease_in_out_elastic_overshoot(t: f64, overshoot: f64) f64 {
    if (t == 0.0) return 0.0;
    if (t == 1.0) return 1.0;
    const period = 0.3 * 1.5;
    const amp = @max(overshoot, 1.0);
    const s = period / (2.0 * math.pi) * math.asin(1.0 / amp);
    if (t < 0.5) {
        return -0.5 * (amp * math.pow(f64, 2.0, 20.0 * t - 10.0) * math.sin((20.0 * t - 10.0 - s) * (2.0 * math.pi / period)));
    } else {
        return 0.5 * (amp * math.pow(f64, 2.0, -20.0 * t + 10.0) * math.sin((20.0 * t - 10.0 - s) * (2.0 * math.pi / period))) + 1.0;
    }
}

fn bounce_ease_out(t: f64) f64 {
    if (t < 4.0 / 11.0) {
        return (121.0 * t * t) / 16.0;
    } else if (t < 8.0 / 11.0) {
        return (363.0 / 40.0 * t * t) - (99.0 / 10.0 * t) + 17.0 / 5.0;
    } else if (t < 9.0 / 10.0) {
        return (4356.0 / 361.0 * t * t) - (35442.0 / 1805.0 * t) + 16061.0 / 1805.0;
    } else {
        return (54.0 / 5.0 * t * t) - (513.0 / 25.0 * t) + 268.0 / 25.0;
    }
}

pub fn ease_in_bounce(t: f64) f64 {
    return 1.0 - bounce_ease_out(1.0 - t);
}

pub fn ease_out_bounce(t: f64) f64 {
    return bounce_ease_out(t);
}

pub fn ease_in_out_bounce(t: f64) f64 {
    if (t < 0.5) {
        return 0.5 * ease_in_bounce(t * 2.0);
    } else {
        return 0.5 * ease_out_bounce(t * 2.0 - 1.0) + 0.5;
    }
}

test {
    const eps = 1e-9;

    const TestFn = *const fn (f64) f64;
    const endpoints = struct {
        fn check(comptime f: TestFn) !void {
            try std.testing.expectApproxEqAbs(0.0, f(0.0), eps);
            try std.testing.expectApproxEqAbs(1.0, f(1.0), eps);
        }
    }.check;

    try endpoints(linear);

    try endpoints(ease_in_quad);
    try endpoints(ease_out_quad);
    try endpoints(ease_in_out_quad);

    try endpoints(ease_in_cubic);
    try endpoints(ease_out_cubic);
    try endpoints(ease_in_out_cubic);

    try endpoints(ease_in_quart);
    try endpoints(ease_out_quart);
    try endpoints(ease_in_out_quart);

    try endpoints(ease_in_quint);
    try endpoints(ease_out_quint);
    try endpoints(ease_in_out_quint);

    try endpoints(ease_in_sine);
    try endpoints(ease_out_sine);
    try endpoints(ease_in_out_sine);

    try endpoints(ease_in_circ);
    try endpoints(ease_out_circ);
    try endpoints(ease_in_out_circ);

    try endpoints(ease_in_back);
    try endpoints(ease_out_back);
    try endpoints(ease_in_out_back);

    try endpoints(ease_in_elastic);
    try endpoints(ease_out_elastic);
    try endpoints(ease_in_out_elastic);

    try endpoints(ease_in_bounce);
    try endpoints(ease_out_bounce);
    try endpoints(ease_in_out_bounce);
}
