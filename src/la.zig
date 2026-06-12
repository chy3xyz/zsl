const std = @import("std");
const util = @import("util.zig");
const Error = @import("errors.zig").Error;

pub fn Vector(comptime T: type) type {
    _ = util.Float(T);

    return struct {
        data: []T,
        len: usize,
        stride: usize,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, len: usize) Error!Self {
            if (len == 0) return error.InvalidDimension;
            const data = try allocator.alloc(T, len);
            errdefer allocator.free(data);
            @memset(data, 0);
            return .{
                .data = data,
                .len = len,
                .stride = 1,
            };
        }

        pub fn fromSlice(allocator: std.mem.Allocator, slice: []const T) Error!Self {
            if (slice.len == 0) return error.InvalidDimension;
            const data = try allocator.alloc(T, slice.len);
            errdefer allocator.free(data);
            @memcpy(data, slice);
            return .{
                .data = data,
                .len = slice.len,
                .stride = 1,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.data);
            self.data = &[_]T{};
            self.len = 0;
            self.stride = 1;
        }

        pub fn get(self: Self, i: usize) Error!T {
            try util.checkIndex(self.len, i);
            return self.data[i * self.stride];
        }

        pub fn set(self: Self, i: usize, value: T) Error!void {
            try util.checkIndex(self.len, i);
            self.data[i * self.stride] = value;
        }

        pub fn rawData(self: Self) []T {
            return self.data;
        }
    };
}

pub fn Matrix(comptime T: type) type {
    _ = T;
    return struct {};
}

test "Vector init allocates and deinit frees" {
    const V = Vector(f64);
    var v = try V.init(std.testing.allocator, 3);
    defer v.deinit(std.testing.allocator);
    try std.testing.expectEqual(3, v.len);
    try std.testing.expectEqual(1, v.stride);
}

test "Vector fromSlice copies data" {
    const V = Vector(f32);
    const src = &[_]f32{ 1.0, 2.0, 3.0 };
    var v = try V.fromSlice(std.testing.allocator, src);
    defer v.deinit(std.testing.allocator);
    try std.testing.expectEqual(3, v.len);
    try std.testing.expectEqual(1.0, try v.get(0));
    try std.testing.expectEqual(2.0, try v.get(1));
    try std.testing.expectEqual(3.0, try v.get(2));
}

test "Vector get/set with stride" {
    const V = Vector(f32);
    const src = &[_]f32{ 1.0, 0.0, 2.0, 0.0, 3.0 };
    var v = try V.fromSlice(std.testing.allocator, src);
    defer v.deinit(std.testing.allocator);
    v.stride = 2;
    try std.testing.expectEqual(1.0, try v.get(0));
    try std.testing.expectEqual(2.0, try v.get(1));
    try std.testing.expectEqual(3.0, try v.get(2));
}

test "Vector bounds check returns error" {
    const V = Vector(f64);
    var v = try V.init(std.testing.allocator, 2);
    defer v.deinit(std.testing.allocator);
    try std.testing.expectError(error.IndexOutOfBounds, v.get(2));
}
