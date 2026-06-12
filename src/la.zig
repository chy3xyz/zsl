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
    _ = util.Float(T);

    return struct {
        data: []T,
        rows: usize,
        cols: usize,
        row_stride: usize,
        col_stride: usize,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) Error!Self {
            if (rows == 0 or cols == 0) return error.InvalidDimension;
            const data = try allocator.alloc(T, rows * cols);
            errdefer allocator.free(data);
            @memset(data, 0);
            return .{
                .data = data,
                .rows = rows,
                .cols = cols,
                .row_stride = cols,
                .col_stride = 1,
            };
        }

        pub fn fromRowSlice(allocator: std.mem.Allocator, rows: usize, cols: usize, slice: []const T) Error!Self {
            if (rows == 0 or cols == 0) return error.InvalidDimension;
            if (slice.len != rows * cols) return error.ShapeMismatch;
            const data = try allocator.alloc(T, slice.len);
            errdefer allocator.free(data);
            @memcpy(data, slice);
            return .{
                .data = data,
                .rows = rows,
                .cols = cols,
                .row_stride = cols,
                .col_stride = 1,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.data);
            self.data = &[_]T{};
            self.rows = 0;
            self.cols = 0;
            self.row_stride = 1;
            self.col_stride = 1;
        }

        pub fn get(self: Self, r: usize, c: usize) Error!T {
            try util.checkIndex(self.rows, r);
            try util.checkIndex(self.cols, c);
            return self.data[r * self.row_stride + c * self.col_stride];
        }

        pub fn set(self: Self, r: usize, c: usize, value: T) Error!void {
            try util.checkIndex(self.rows, r);
            try util.checkIndex(self.cols, c);
            self.data[r * self.row_stride + c * self.col_stride] = value;
        }

        pub fn row(self: Self, r: usize) Error!Vector(T) {
            try util.checkIndex(self.rows, r);
            return .{
                .data = self.data[r * self.row_stride ..],
                .len = self.cols,
                .stride = self.col_stride,
            };
        }

        pub fn col(self: Self, c: usize) Error!Vector(T) {
            try util.checkIndex(self.cols, c);
            return .{
                .data = self.data[c * self.col_stride ..],
                .len = self.rows,
                .stride = self.row_stride,
            };
        }

        pub fn transpose(self: Self) Self {
            return .{
                .data = self.data,
                .rows = self.cols,
                .cols = self.rows,
                .row_stride = self.col_stride,
                .col_stride = self.row_stride,
            };
        }

        pub fn rawData(self: Self) []T {
            return self.data;
        }
    };
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

test "Matrix init shape" {
    const M = Matrix(f64);
    var m = try M.init(std.testing.allocator, 2, 3);
    defer m.deinit(std.testing.allocator);
    try std.testing.expectEqual(2, m.rows);
    try std.testing.expectEqual(3, m.cols);
}

test "Matrix get/set row-major" {
    const M = Matrix(f64);
    var m = try M.init(std.testing.allocator, 2, 3);
    defer m.deinit(std.testing.allocator);
    try m.set(0, 0, 1.0);
    try m.set(0, 1, 2.0);
    try m.set(0, 2, 3.0);
    try m.set(1, 0, 4.0);
    try m.set(1, 1, 5.0);
    try m.set(1, 2, 6.0);
    try std.testing.expectEqual(1.0, try m.get(0, 0));
    try std.testing.expectEqual(6.0, try m.get(1, 2));
}

test "Matrix fromRowSlice" {
    const M = Matrix(f32);
    const src = &[_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var m = try M.fromRowSlice(std.testing.allocator, 2, 2, src);
    defer m.deinit(std.testing.allocator);
    try std.testing.expectEqual(1.0, try m.get(0, 0));
    try std.testing.expectEqual(4.0, try m.get(1, 1));
}

test "Matrix row/col views" {
    const M = Matrix(f64);
    var m = try M.init(std.testing.allocator, 2, 3);
    defer m.deinit(std.testing.allocator);
    try m.set(0, 0, 1.0);
    try m.set(0, 1, 2.0);
    try m.set(0, 2, 3.0);
    try m.set(1, 0, 4.0);
    try m.set(1, 1, 5.0);
    try m.set(1, 2, 6.0);

    var row0 = try m.row(0);
    try std.testing.expectEqual(1.0, try row0.get(0));
    try std.testing.expectEqual(2.0, try row0.get(1));

    var col1 = try m.col(1);
    try std.testing.expectEqual(2.0, try col1.get(0));
    try std.testing.expectEqual(5.0, try col1.get(1));
}

test "Matrix transpose is a view" {
    const M = Matrix(f64);
    var m = try M.init(std.testing.allocator, 2, 3);
    defer m.deinit(std.testing.allocator);
    try m.set(0, 2, 7.0);
    const mt = m.transpose();
    try std.testing.expectEqual(7.0, try mt.get(2, 0));
    try std.testing.expectEqual(3, mt.rows);
    try std.testing.expectEqual(2, mt.cols);
}
