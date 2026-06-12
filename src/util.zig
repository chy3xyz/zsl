const std = @import("std");

pub fn isFloat(comptime T: type) bool {
    return @typeInfo(T) == .float;
}

pub fn Float(comptime T: type) type {
    if (!isFloat(T)) {
        @compileError("Expected a floating-point type, found " ++ @typeName(T));
    }
    return T;
}

pub fn checkIndex(len: usize, i: usize) error{IndexOutOfBounds}!void {
    if (i >= len) return error.IndexOutOfBounds;
}

pub fn checkSameLength(a: usize, b: usize) error{ShapeMismatch}!void {
    if (a != b) return error.ShapeMismatch;
}

test "isFloat accepts floating-point types" {
    try std.testing.expect(isFloat(f16));
    try std.testing.expect(isFloat(f32));
    try std.testing.expect(isFloat(f64));
    try std.testing.expect(isFloat(f128));
    try std.testing.expect(!isFloat(u32));
    try std.testing.expect(!isFloat(i32));
    try std.testing.expect(!isFloat(bool));
}

test "Float constraint returns the input type for floats" {
    try std.testing.expect(Float(f32) == f32);
    try std.testing.expect(Float(f64) == f64);
}
