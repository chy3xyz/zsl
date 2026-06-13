const std = @import("std");
const la = @import("../la.zig");
const Error = @import("../errors.zig").Error;

/// `Data(T)` holds ML data in matrix format.
///
///   x -- [nb_samples][nb_features] matrix of observations
///   y -- [nb_samples] optional target vector
pub fn Data(comptime T: type) type {
    return struct {
        nb_samples: usize,
        nb_features: usize,
        x: la.Matrix(T),
        y: []T,

        const Self = @This();

        /// Returns a new object to hold ML data.
        ///
        /// If `allocate` is true, allocates `x` (and `y` when `use_y` is true);
        /// otherwise the caller must set `x` and `y` with `set()`.
        pub fn init(
            allocator: std.mem.Allocator,
            nb_samples: usize,
            nb_features: usize,
            use_y: bool,
            allocate: bool,
        ) Error!Self {
            if (nb_samples == 0 or nb_features == 0) return error.InvalidDimension;

            var x: la.Matrix(T) = undefined;
            var y: []T = &[_]T{};

            if (allocate) {
                x = try la.Matrix(T).init(allocator, nb_samples, nb_features);
                errdefer x.deinit(allocator);
                if (use_y) {
                    y = try allocator.alloc(T, nb_samples);
                    errdefer allocator.free(y);
                    @memset(y, 0);
                }
            } else {
                x = .{
                    .data = &[_]T{},
                    .rows = 0,
                    .cols = 0,
                    .row_stride = 1,
                    .col_stride = 1,
                };
            }

            return .{
                .nb_samples = nb_samples,
                .nb_features = nb_features,
                .x = x,
                .y = y,
            };
        }

        /// Constructs a `Data(T)` from a jagged slice of rows.
        ///
        /// All rows must have the same length and at least one row must be provided.
        pub fn fromRawX(
            allocator: std.mem.Allocator,
            xraw: []const []const T,
        ) Error!Self {
            if (xraw.len == 0) return error.InvalidDimension;
            const nb_samples = xraw.len;
            const nb_features = xraw[0].len;
            if (nb_features == 0) return error.InvalidDimension;

            for (xraw[1..]) |row| {
                if (row.len != nb_features) return error.ShapeMismatch;
            }

            var self = try init(allocator, nb_samples, nb_features, true, true);
            errdefer self.deinit(allocator);

            for (0..nb_samples) |i| {
                for (0..nb_features) |j| {
                    try self.x.set(i, j, xraw[i][j]);
                }
            }

            return self;
        }

        /// Constructs a `Data(T)` from a jagged slice of rows and a target vector.
        ///
        /// `xraw` must contain at least one row, all rows must be rectangular, and
        /// `yraw.len` must equal `xraw.len`.
        pub fn fromRawXy(
            allocator: std.mem.Allocator,
            xraw: []const []const T,
            yraw: []const T,
        ) Error!Self {
            if (xraw.len == 0) return error.InvalidDimension;
            const nb_samples = xraw.len;
            if (yraw.len != nb_samples) return error.ShapeMismatch;
            const nb_features = xraw[0].len;
            if (nb_features == 0) return error.InvalidDimension;

            for (xraw[1..]) |row| {
                if (row.len != nb_features) return error.ShapeMismatch;
            }

            var self = try init(allocator, nb_samples, nb_features, false, true);
            errdefer self.deinit(allocator);

            for (0..nb_samples) |i| {
                for (0..nb_features) |j| {
                    try self.x.set(i, j, xraw[i][j]);
                }
            }

            self.y = try allocator.alloc(T, nb_samples);
            errdefer allocator.free(self.y);
            @memcpy(self.y, yraw);

            return self;
        }

        /// Sets the `x` matrix and optional `y` vector, updating dimensions from `x`.
        pub fn set(self: *Self, x: la.Matrix(T), y: []T) Error!void {
            if (y.len < x.rows) return error.ShapeMismatch;
            self.nb_samples = x.rows;
            self.nb_features = x.cols;
            self.x = x;
            self.y = y;
        }

        /// Sets the `x` matrix, updating dimensions from `x`.
        pub fn setX(self: *Self, x: la.Matrix(T)) Error!void {
            if (x.rows < self.nb_samples or x.cols < self.nb_features) return error.ShapeMismatch;
            self.x = x;
            self.nb_samples = x.rows;
            self.nb_features = x.cols;
        }

        /// Sets the `y` vector.
        pub fn setY(self: *Self, y: []T) Error!void {
            if (y.len < self.nb_samples) return error.ShapeMismatch;
            self.y = y;
        }

        /// Releases all owned memory.
        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            if (self.x.data.len > 0) {
                self.x.deinit(allocator);
            }
            if (self.y.len > 0) {
                allocator.free(self.y);
            }
            self.nb_samples = 0;
            self.nb_features = 0;
            self.y = &[_]T{};
        }
    };
}

test "Data init allocates x and optional y" {
    const D = Data(f64);
    var d = try D.init(std.testing.allocator, 3, 2, true, true);
    defer d.deinit(std.testing.allocator);
    try std.testing.expectEqual(3, d.nb_samples);
    try std.testing.expectEqual(2, d.nb_features);
    try std.testing.expectEqual(3, d.x.rows);
    try std.testing.expectEqual(2, d.x.cols);
    try std.testing.expectEqual(3, d.y.len);
}

test "Data fromRawX copies jagged slice" {
    const D = Data(f64);
    const xraw = &[_][]const f64{
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 3.0, 4.0 },
        &[_]f64{ 5.0, 6.0 },
    };
    var d = try D.fromRawX(std.testing.allocator, xraw);
    defer d.deinit(std.testing.allocator);
    try std.testing.expectEqual(3, d.nb_samples);
    try std.testing.expectEqual(2, d.nb_features);
    try std.testing.expectApproxEqAbs(4.0, try d.x.get(1, 1), 1e-12);
    try std.testing.expectEqual(3, d.y.len);
}

test "Data fromRawXy copies x and y" {
    const D = Data(f64);
    const xraw = &[_][]const f64{
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 3.0, 4.0 },
    };
    const yraw = &[_]f64{ 0.0, 1.0 };
    var d = try D.fromRawXy(std.testing.allocator, xraw, yraw);
    defer d.deinit(std.testing.allocator);
    try std.testing.expectEqual(2, d.nb_samples);
    try std.testing.expectEqual(2, d.nb_features);
    try std.testing.expectApproxEqAbs(1.0, d.y[1], 1e-12);
}

test "Data setters update fields" {
    const D = Data(f64);
    var d = try D.init(std.testing.allocator, 2, 2, true, true);
    defer d.deinit(std.testing.allocator);

    var new_x = try la.Matrix(f64).fromRowSlice(std.testing.allocator, 3, 3, &[_]f64{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    });
    errdefer new_x.deinit(std.testing.allocator);

    const new_y = try std.testing.allocator.alloc(f64, 3);
    errdefer std.testing.allocator.free(new_y);
    @memset(new_y, 0.0);

    var old_x = d.x;
    const old_y = d.y;

    try d.set(new_x, new_y);

    old_x.deinit(std.testing.allocator);
    std.testing.allocator.free(old_y);

    try std.testing.expectEqual(3, d.nb_samples);
    try std.testing.expectEqual(3, d.nb_features);
    try std.testing.expectApproxEqAbs(5.0, try d.x.get(1, 1), 1e-12);
}
