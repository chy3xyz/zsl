const std = @import("std");
const util = @import("util.zig");

pub fn eps(comptime T: type) T {
    _ = util.Float(T);
    return std.math.floatEps(T);
}

pub fn approxEqAbs(comptime T: type, a: T, b: T, tol: T) bool {
    _ = util.Float(T);
    return @abs(a - b) <= tol;
}

pub fn approxEqRel(comptime T: type, a: T, b: T, rel_tol: T, abs_tol: T) bool {
    _ = util.Float(T);
    const diff = @abs(a - b);
    const largest = @max(@abs(a), @abs(b));
    return diff <= largest * rel_tol or diff <= abs_tol;
}

pub fn isFinite(comptime T: type, x: T) bool {
    _ = util.Float(T);
    return std.math.isFinite(x);
}

test "eps returns machine epsilon" {
    try std.testing.expect(eps(f32) > 0);
    try std.testing.expect(eps(f64) > 0);
    try std.testing.expectApproxEqAbs(eps(f32), 1.1920929e-7, 1e-10);
    try std.testing.expectApproxEqAbs(eps(f64), 2.220446049250313e-16, 1e-20);
}

test "approxEqAbs works for f32 and f64" {
    try std.testing.expect(approxEqAbs(f32, 1.0, 1.000001, 1e-5));
    try std.testing.expect(!approxEqAbs(f32, 1.0, 1.0001, 1e-5));
    try std.testing.expect(approxEqAbs(f64, 1.0, 1.000000001, 1e-8));
}

test "approxEqRel works near zero and away from zero" {
    try std.testing.expect(approxEqRel(f64, 1e-12, 2e-12, 1e-9, 1e-9));
    try std.testing.expect(approxEqRel(f64, 1e6, 1e6 + 1, 1e-5, 1e-9));
}

test "isFinite rejects infinities and NaN" {
    try std.testing.expect(isFinite(f32, 1.0));
    try std.testing.expect(!isFinite(f32, std.math.inf(f32)));
    try std.testing.expect(!isFinite(f32, std.math.nan(f32)));
}
