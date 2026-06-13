const std = @import("std");
const Error = @import("errors.zig").Error;

pub const Complex = struct {
    re: f64,
    im: f64,
};

pub fn complex(re: f64, im: f64) Complex {
    return .{ .re = re, .im = im };
}

pub fn add(a: Complex, b: Complex) Complex {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}

pub fn sub(a: Complex, b: Complex) Complex {
    return .{ .re = a.re - b.re, .im = a.im - b.im };
}

pub fn mul(a: Complex, b: Complex) Complex {
    return .{
        .re = a.re * b.re - a.im * b.im,
        .im = a.re * b.im + a.im * b.re,
    };
}

pub fn scale(a: Complex, s: f64) Complex {
    return .{ .re = a.re * s, .im = a.im * s };
}

pub fn conj(a: Complex) Complex {
    return .{ .re = a.re, .im = -a.im };
}

pub fn abs(a: Complex) f64 {
    return std.math.hypot(a.re, a.im);
}

pub fn arg(a: Complex) f64 {
    return std.math.atan2(a.im, a.re);
}

pub fn exp(a: Complex) Complex {
    const r = std.math.exp(a.re);
    return .{
        .re = r * std.math.cos(a.im),
        .im = r * std.math.sin(a.im),
    };
}

fn isPowerOfTwo(n: usize) bool {
    return n != 0 and (n & (n - 1)) == 0;
}

pub fn next_power_of_2(n: usize) usize {
    if (n == 0) return 1;
    var p: usize = 1;
    while (p < n) p <<= 1;
    return p;
}

fn fftInPlace(buf: []Complex, comptime sign: f64) void {
    const n = buf.len;

    var j: usize = 0;
    for (1..n) |i| {
        var bit = n >> 1;
        while (j & bit != 0) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            const tmp = buf[i];
            buf[i] = buf[j];
            buf[j] = tmp;
        }
    }

    var len: usize = 2;
    while (len <= n) : (len <<= 1) {
        const ang = sign * 2.0 * std.math.pi / @as(f64, @floatFromInt(len));
        const wlen = complex(@cos(ang), @sin(ang));
        var i: usize = 0;
        while (i < n) : (i += len) {
            var w = complex(1.0, 0.0);
            const half = len >> 1;
            for (0..half) |k| {
                const u = buf[i + k];
                const v = mul(buf[i + k + half], w);
                buf[i + k] = add(u, v);
                buf[i + k + half] = sub(u, v);
                w = mul(w, wlen);
            }
        }
    }
}

pub fn fft_complex(input: []const Complex, allocator: std.mem.Allocator) Error![]Complex {
    const n = input.len;
    if (n < 2 or !isPowerOfTwo(n)) return error.InvalidDimension;

    const out = try allocator.alloc(Complex, n);
    errdefer allocator.free(out);
    @memcpy(out, input);

    fftInPlace(out, -1.0);
    return out;
}

pub fn ifft(input: []const Complex, allocator: std.mem.Allocator) Error![]Complex {
    const n = input.len;
    if (n < 2 or !isPowerOfTwo(n)) return error.InvalidDimension;

    const out = try allocator.alloc(Complex, n);
    errdefer allocator.free(out);
    @memcpy(out, input);

    fftInPlace(out, 1.0);
    return out;
}

pub fn fft(input: []const f64, allocator: std.mem.Allocator) Error![]Complex {
    const n = input.len;
    if (n < 2 or !isPowerOfTwo(n)) return error.InvalidDimension;

    const out = try allocator.alloc(Complex, n);
    errdefer allocator.free(out);
    for (input, out) |v, *c| {
        c.* = complex(v, 0.0);
    }

    fftInPlace(out, -1.0);
    return out;
}

pub fn ifft_normalized(input: []const Complex, allocator: std.mem.Allocator) Error![]f64 {
    const n = input.len;
    if (n < 2 or !isPowerOfTwo(n)) return error.InvalidDimension;

    const tmp = try ifft(input, allocator);
    defer allocator.free(tmp);

    const out = try allocator.alloc(f64, n);
    errdefer allocator.free(out);
    const inv_n = 1.0 / @as(f64, @floatFromInt(n));
    for (tmp, out) |c, *r| {
        r.* = c.re * inv_n;
    }
    return out;
}

pub fn fftshift(input: []const Complex, allocator: std.mem.Allocator) Error![]Complex {
    const n = input.len;
    if (n == 0) return error.InvalidDimension;

    const out = try allocator.alloc(Complex, n);
    errdefer allocator.free(out);
    const shift = n >> 1;
    for (0..n) |i| {
        out[i] = input[(i + shift) % n];
    }
    return out;
}

pub fn ifftshift(input: []const Complex, allocator: std.mem.Allocator) Error![]Complex {
    const n = input.len;
    if (n == 0) return error.InvalidDimension;

    const out = try allocator.alloc(Complex, n);
    errdefer allocator.free(out);
    const shift = (n + 1) >> 1;
    for (0..n) |i| {
        out[i] = input[(i + shift) % n];
    }
    return out;
}

pub fn magnitude_spectrum(input: []const Complex, allocator: std.mem.Allocator) Error![]f64 {
    const out = try allocator.alloc(f64, input.len);
    errdefer allocator.free(out);
    for (input, out) |c, *r| {
        r.* = abs(c);
    }
    return out;
}

pub fn power_spectrum(input: []const Complex, allocator: std.mem.Allocator) Error![]f64 {
    const out = try allocator.alloc(f64, input.len);
    errdefer allocator.free(out);
    for (input, out) |c, *r| {
        r.* = c.re * c.re + c.im * c.im;
    }
    return out;
}

test "complex arithmetic helpers" {
    const a = complex(1.0, 2.0);
    const b = complex(3.0, 4.0);

    const s = add(a, b);
    try std.testing.expectEqual(4.0, s.re);
    try std.testing.expectEqual(6.0, s.im);

    const d = sub(a, b);
    try std.testing.expectEqual(-2.0, d.re);
    try std.testing.expectEqual(-2.0, d.im);

    const p = mul(a, b);
    try std.testing.expectEqual(-5.0, p.re);
    try std.testing.expectEqual(10.0, p.im);

    const sc = scale(a, 2.0);
    try std.testing.expectEqual(2.0, sc.re);
    try std.testing.expectEqual(4.0, sc.im);

    const cj = conj(a);
    try std.testing.expectEqual(1.0, cj.re);
    try std.testing.expectEqual(-2.0, cj.im);

    try std.testing.expectApproxEqAbs(std.math.sqrt(5.0), abs(a), 1e-12);
    try std.testing.expectApproxEqAbs(5.0, abs(complex(3.0, 4.0)), 1e-12);
    try std.testing.expectApproxEqAbs(std.math.pi / 4.0, arg(complex(1.0, 1.0)), 1e-12);

    const e = exp(complex(0.0, std.math.pi));
    try std.testing.expectApproxEqAbs(-1.0, e.re, 1e-12);
    try std.testing.expectApproxEqAbs(0.0, e.im, 1e-12);
}

test "fft rejects invalid dimensions" {
    const input = &[_]f64{ 1.0, 2.0, 3.0 };
    try std.testing.expectError(error.InvalidDimension, fft(input, std.testing.allocator));
    try std.testing.expectError(error.InvalidDimension, fft(&[_]f64{1.0}, std.testing.allocator));
}

test "fft of impulse is all ones" {
    const input = &[_]f64{ 1.0, 0.0, 0.0, 0.0 };
    const out = try fft(input, std.testing.allocator);
    defer std.testing.allocator.free(out);

    try std.testing.expectEqual(4, out.len);
    for (out) |c| {
        try std.testing.expectApproxEqAbs(1.0, c.re, 1e-12);
        try std.testing.expectApproxEqAbs(0.0, c.im, 1e-12);
    }
}

test "fft of single frequency sinusoid peaks at expected bins" {
    const n = 8;
    var input: [n]f64 = undefined;
    const k: f64 = 1.0;
    for (0..n) |i| {
        const t = 2.0 * std.math.pi * k * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n));
        input[i] = @cos(t);
    }

    const out = try fft(&input, std.testing.allocator);
    defer std.testing.allocator.free(out);

    const mag = try magnitude_spectrum(out, std.testing.allocator);
    defer std.testing.allocator.free(mag);

    for (mag, 0..) |m, i| {
        if (i == 1 or i == n - 1) {
            try std.testing.expectApproxEqAbs(@as(f64, @floatFromInt(n)) / 2.0, m, 1e-9);
        } else {
            try std.testing.expectApproxEqAbs(0.0, m, 1e-9);
        }
    }
}

test "fft round-trip via ifft_normalized" {
    const input = &[_]f64{ 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0 };
    const spectrum = try fft(input, std.testing.allocator);
    defer std.testing.allocator.free(spectrum);

    const recovered = try ifft_normalized(spectrum, std.testing.allocator);
    defer std.testing.allocator.free(recovered);

    for (input, recovered) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 1e-12);
    }
}

test "fftshift and ifftshift are inverses" {
    const input = &[_]Complex{
        complex(0.0, 0.0),
        complex(1.0, 0.0),
        complex(2.0, 0.0),
        complex(3.0, 0.0),
        complex(4.0, 0.0),
    };
    const shifted = try fftshift(input, std.testing.allocator);
    defer std.testing.allocator.free(shifted);
    const unshifted = try ifftshift(shifted, std.testing.allocator);
    defer std.testing.allocator.free(unshifted);

    for (input, unshifted) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected.re, actual.re, 1e-12);
        try std.testing.expectApproxEqAbs(expected.im, actual.im, 1e-12);
    }
}

test "magnitude and power spectrum" {
    const input = &[_]Complex{ complex(3.0, 4.0), complex(0.0, 1.0) };
    const mag = try magnitude_spectrum(input, std.testing.allocator);
    defer std.testing.allocator.free(mag);
    const pow = try power_spectrum(input, std.testing.allocator);
    defer std.testing.allocator.free(pow);

    try std.testing.expectApproxEqAbs(5.0, mag[0], 1e-12);
    try std.testing.expectApproxEqAbs(1.0, mag[1], 1e-12);
    try std.testing.expectApproxEqAbs(25.0, pow[0], 1e-12);
    try std.testing.expectApproxEqAbs(1.0, pow[1], 1e-12);
}

test "next_power_of_2" {
    try std.testing.expectEqual(@as(usize, 1), next_power_of_2(0));
    try std.testing.expectEqual(@as(usize, 1), next_power_of_2(1));
    try std.testing.expectEqual(@as(usize, 2), next_power_of_2(2));
    try std.testing.expectEqual(@as(usize, 4), next_power_of_2(3));
    try std.testing.expectEqual(@as(usize, 8), next_power_of_2(5));
    try std.testing.expectEqual(@as(usize, 16), next_power_of_2(16));
}
