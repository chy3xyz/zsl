const std = @import("std");
const Error = @import("errors.zig").Error;

/// Evaluates a polynomial `P(x) = c[0] + c[1]*x + ... + c[n]*x^n`
/// using Horner's method.
pub fn eval(c: []const f64, x: f64) Error!f64 {
    if (c.len == 0) return error.InvalidDimension;
    var ans = c[c.len - 1];
    var i: usize = c.len - 1;
    while (i > 0) {
        i -= 1;
        ans = c[i] + x * ans;
    }
    return ans;
}

/// Evaluates a polynomial and its derivatives at `x`.
/// Returns `[P(x), P'(x), ..., P^(lenres-1)(x)]`.
pub fn eval_derivs(c: []const f64, x: f64, lenres: usize, allocator: std.mem.Allocator) Error![]f64 {
    if (c.len == 0) return error.InvalidDimension;
    if (lenres == 0) return allocator.alloc(f64, 0);

    const res = try allocator.alloc(f64, lenres);
    errdefer allocator.free(res);
    @memset(res, 0.0);

    var i: isize = @intCast(c.len - 1);
    while (i >= 0) : (i -= 1) {
        const ci = c[@intCast(i)];
        var j: usize = lenres - 1;
        while (j > 0) : (j -= 1) {
            res[j] = res[j] * x + res[j - 1];
        }
        res[0] = res[0] * x + ci;
    }

    var f: f64 = 1.0;
    for (2..lenres) |k| {
        f *= @floatFromInt(k);
        res[k] *= f;
    }

    return res;
}

/// Solves `a*x^2 + b*x + c = 0` for real roots.
pub fn solve_quadratic(a: f64, b: f64, c: f64, allocator: std.mem.Allocator) Error![]f64 {
    if (a == 0.0) {
        if (b == 0.0) {
            return allocator.alloc(f64, 0);
        }
        const roots = try allocator.alloc(f64, 1);
        roots[0] = -c / b;
        return roots;
    }

    const disc = b * b - 4.0 * a * c;
    if (disc > 0.0) {
        const roots = try allocator.alloc(f64, 2);
        if (b == 0.0) {
            const r = std.math.sqrt(-c / a);
            roots[0] = -r;
            roots[1] = r;
        } else {
            const sgnb: f64 = if (b > 0.0) 1.0 else -1.0;
            const temp = -0.5 * (b + sgnb * std.math.sqrt(disc));
            var r1 = temp / a;
            var r2 = c / temp;
            if (r1 > r2) {
                const tmp = r1;
                r1 = r2;
                r2 = tmp;
            }
            roots[0] = r1;
            roots[1] = r2;
        }
        return roots;
    } else if (disc == 0.0) {
        const roots = try allocator.alloc(f64, 2);
        roots[0] = -0.5 * b / a;
        roots[1] = roots[0];
        return roots;
    } else {
        return allocator.alloc(f64, 0);
    }
}

fn sorted_3_(x_: f64, y_: f64, z_: f64) struct { f64, f64, f64 } {
    var x = x_;
    var y = y_;
    var z = z_;
    if (x > y) {
        const t = x;
        x = y;
        y = t;
    }
    if (y > z) {
        const t = y;
        y = z;
        z = t;
    }
    if (x > y) {
        const t = x;
        x = y;
        y = t;
    }
    return .{ x, y, z };
}

/// Solves `a*x^3 + b*x^2 + c*x + d = 0` for real roots.
/// Degenerate cases (`a == 0`) are forwarded to `solve_quadratic`.
pub fn solve_cubic(a: f64, b: f64, c: f64, d: f64, allocator: std.mem.Allocator) Error![]f64 {
    if (a == 0.0) {
        return solve_quadratic(b, c, d, allocator);
    }

    // Normalize to monic form: x^3 + A*x^2 + B*x + C = 0
    const A = b / a;
    const B = c / a;
    const C = d / a;

    const q_ = A * A - 3.0 * B;
    const r_ = 2.0 * A * A * A - 9.0 * A * B + 27.0 * C;
    const q = q_ / 9.0;
    const r = r_ / 54.0;
    const q3 = q * q * q;
    const r2 = r * r;
    const cr2 = 729.0 * r_ * r_;
    const cq3 = 2916.0 * q_ * q_ * q_;

    const shift = -A / 3.0;

    if (r == 0.0 and q == 0.0) {
        const roots = try allocator.alloc(f64, 3);
        roots[0] = shift;
        roots[1] = shift;
        roots[2] = shift;
        return roots;
    } else if (cr2 == cq3) {
        const sqrt_q = std.math.sqrt(q);
        const roots = try allocator.alloc(f64, 3);
        if (r > 0.0) {
            roots[0] = -2.0 * sqrt_q + shift;
            roots[1] = sqrt_q + shift;
            roots[2] = sqrt_q + shift;
        } else {
            roots[0] = -sqrt_q + shift;
            roots[1] = -sqrt_q + shift;
            roots[2] = 2.0 * sqrt_q + shift;
        }
        return roots;
    } else if (r2 < q3) {
        const sgnr: f64 = if (r >= 0.0) 1.0 else -1.0;
        const ratio = sgnr * std.math.sqrt(r2 / q3);
        const theta = std.math.acos(ratio);
        const norm = -2.0 * std.math.sqrt(q);
        const x0 = norm * std.math.cos(theta / 3.0) + shift;
        const x1 = norm * std.math.cos((theta + 2.0 * std.math.pi) / 3.0) + shift;
        const x2 = norm * std.math.cos((theta - 2.0 * std.math.pi) / 3.0) + shift;
        const s = sorted_3_(x0, x1, x2);
        const roots = try allocator.alloc(f64, 3);
        roots[0] = s[0];
        roots[1] = s[1];
        roots[2] = s[2];
        return roots;
    } else {
        const sgnr: f64 = if (r >= 0.0) 1.0 else -1.0;
        const a_ = -sgnr * std.math.pow(f64, @abs(r) + std.math.sqrt(r2 - q3), 1.0 / 3.0);
        const b_ = q / a_;
        const roots = try allocator.alloc(f64, 1);
        roots[0] = a_ + b_ + shift;
        return roots;
    }
}

// ---------------------------------------------------------------------------
// Additional polynomial helpers
// ---------------------------------------------------------------------------

/// Adds two polynomials: `(a_n*x^n + ... + a_0) + (b_m*x^m + ... + b_0)`.
pub fn add(a: []const f64, b: []const f64, allocator: std.mem.Allocator) Error![]f64 {
    const max_len = @max(a.len, b.len);
    const result = try allocator.alloc(f64, max_len);
    errdefer allocator.free(result);
    for (0..max_len) |i| {
        const av = if (i < a.len) a[i] else 0.0;
        const bv = if (i < b.len) b[i] else 0.0;
        result[i] = av + bv;
    }
    return result;
}

/// Subtracts two polynomials: `(a_n*x^n + ... + a_0) - (b_m*x^m + ... + b_0)`.
pub fn subtract(a: []const f64, b: []const f64, allocator: std.mem.Allocator) Error![]f64 {
    const max_len = @max(a.len, b.len);
    const result = try allocator.alloc(f64, max_len);
    errdefer allocator.free(result);
    for (0..max_len) |i| {
        const av = if (i < a.len) a[i] else 0.0;
        const bv = if (i < b.len) b[i] else 0.0;
        result[i] = av - bv;
    }
    return result;
}

/// Multiplies two polynomials.
pub fn multiply(a: []const f64, b: []const f64, allocator: std.mem.Allocator) Error![]f64 {
    if (a.len == 0 or b.len == 0) return error.InvalidDimension;
    const result_len = a.len + b.len - 1;
    const result = try allocator.alloc(f64, result_len);
    errdefer allocator.free(result);
    @memset(result, 0.0);
    for (0..a.len) |i| {
        for (0..b.len) |j| {
            result[i + j] += a[i] * b[j];
        }
    }
    return result;
}

/// Returns the degree of a polynomial. Empty coefficient slice is an error.
pub fn degree(c: []const f64) Error!usize {
    if (c.len == 0) return error.InvalidDimension;
    return c.len - 1;
}

/// Sum of coefficients at odd indices (`c[1] + c[3] + ...`).
pub fn sum_odd_coeffs(c: []const f64) f64 {
    var sum: f64 = 0.0;
    var i: usize = 1;
    while (i < c.len) : (i += 2) {
        sum += c[i];
    }
    return sum;
}

/// Sum of coefficients at even indices (`c[0] + c[2] + ...`).
pub fn sum_even_coeffs(c: []const f64) f64 {
    var sum: f64 = 0.0;
    var i: usize = 0;
    while (i < c.len) : (i += 2) {
        sum += c[i];
    }
    return sum;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "eval basic" {
    const c = &[_]f64{ 1.0, -2.0, 1.0 }; // P(x) = x^2 - 2x + 1
    try std.testing.expectApproxEqAbs(try eval(c, 0.0), 1.0, 1e-12);
    try std.testing.expectApproxEqAbs(try eval(c, 1.0), 0.0, 1e-12);
    try std.testing.expectApproxEqAbs(try eval(c, 2.0), 1.0, 1e-12);
    try std.testing.expectApproxEqAbs(try eval(c, 3.0), 4.0, 1e-12);
}

test "eval empty coefficients returns InvalidDimension" {
    const c = &[_]f64{};
    try std.testing.expectError(error.InvalidDimension, eval(c, 1.0));
}

test "eval_derivs basic" {
    // P(x) = x^2 - 2x + 1, P'(x) = 2x - 2, P''(x) = 2
    const c = &[_]f64{ 1.0, -2.0, 1.0 };
    const res = try eval_derivs(c, 2.0, 3, std.testing.allocator);
    defer std.testing.allocator.free(res);
    try std.testing.expectApproxEqAbs(res[0], 1.0, 1e-12);
    try std.testing.expectApproxEqAbs(res[1], 2.0, 1e-12);
    try std.testing.expectApproxEqAbs(res[2], 2.0, 1e-12);
}

test "eval_derivs returns zero for higher derivatives" {
    const c = &[_]f64{ 5.0, -3.0 }; // P(x) = -3x + 5
    const res = try eval_derivs(c, 10.0, 4, std.testing.allocator);
    defer std.testing.allocator.free(res);
    try std.testing.expectApproxEqAbs(res[0], -25.0, 1e-12);
    try std.testing.expectApproxEqAbs(res[1], -3.0, 1e-12);
    try std.testing.expectApproxEqAbs(res[2], 0.0, 1e-12);
    try std.testing.expectApproxEqAbs(res[3], 0.0, 1e-12);
}

test "solve_quadratic basic" {
    // x^2 - 5x + 6 = 0 -> roots 2, 3
    const roots = try solve_quadratic(1.0, -5.0, 6.0, std.testing.allocator);
    defer std.testing.allocator.free(roots);
    try std.testing.expectEqual(@as(usize, 2), roots.len);
    try std.testing.expectApproxEqAbs(roots[0], 2.0, 1e-12);
    try std.testing.expectApproxEqAbs(roots[1], 3.0, 1e-12);
}

test "solve_quadratic no real roots" {
    const roots = try solve_quadratic(1.0, 0.0, 1.0, std.testing.allocator);
    defer std.testing.allocator.free(roots);
    try std.testing.expectEqual(@as(usize, 0), roots.len);
}

test "solve_quadratic linear fallback" {
    const roots = try solve_quadratic(0.0, 2.0, -4.0, std.testing.allocator);
    defer std.testing.allocator.free(roots);
    try std.testing.expectEqual(@as(usize, 1), roots.len);
    try std.testing.expectApproxEqAbs(roots[0], 2.0, 1e-12);
}

test "solve_cubic basic" {
    // x^3 - 6x^2 + 11x - 6 = 0 -> roots 1, 2, 3
    const roots = try solve_cubic(1.0, -6.0, 11.0, -6.0, std.testing.allocator);
    defer std.testing.allocator.free(roots);
    try std.testing.expectEqual(@as(usize, 3), roots.len);
    try std.testing.expectApproxEqAbs(roots[0], 1.0, 1e-12);
    try std.testing.expectApproxEqAbs(roots[1], 2.0, 1e-12);
    try std.testing.expectApproxEqAbs(roots[2], 3.0, 1e-12);
}

test "solve_cubic degenerate quadratic" {
    // 0*x^3 + x^2 - 5x + 6 = 0 -> roots 2, 3
    const roots = try solve_cubic(0.0, 1.0, -5.0, 6.0, std.testing.allocator);
    defer std.testing.allocator.free(roots);
    try std.testing.expectEqual(@as(usize, 2), roots.len);
    try std.testing.expectApproxEqAbs(roots[0], 2.0, 1e-12);
    try std.testing.expectApproxEqAbs(roots[1], 3.0, 1e-12);
}

test "polynomial add and multiply" {
    const p = &[_]f64{ 1.0, 1.0 }; // x + 1
    const q = &[_]f64{ -1.0, 1.0 }; // x - 1

    const sum = try add(p, q, std.testing.allocator);
    defer std.testing.allocator.free(sum);
    try std.testing.expectEqual(@as(usize, 2), sum.len);
    try std.testing.expectApproxEqAbs(sum[0], 0.0, 1e-12);
    try std.testing.expectApproxEqAbs(sum[1], 2.0, 1e-12);

    const prod = try multiply(p, q, std.testing.allocator);
    defer std.testing.allocator.free(prod);
    try std.testing.expectEqual(@as(usize, 3), prod.len);
    try std.testing.expectApproxEqAbs(prod[0], -1.0, 1e-12);
    try std.testing.expectApproxEqAbs(prod[1], 0.0, 1e-12);
    try std.testing.expectApproxEqAbs(prod[2], 1.0, 1e-12);
}

test "degree, sum_odd_coeffs, sum_even_coeffs" {
    const c = &[_]f64{ 1.0, 2.0, 3.0, 4.0 }; // 4 + 3x + 2x^2 + x^3
    try std.testing.expectEqual(@as(usize, 3), try degree(c));
    try std.testing.expectApproxEqAbs(sum_odd_coeffs(c), 6.0, 1e-12);
    try std.testing.expectApproxEqAbs(sum_even_coeffs(c), 4.0, 1e-12);
}
