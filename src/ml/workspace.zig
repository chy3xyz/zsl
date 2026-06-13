const std = @import("std");
const la = @import("../la.zig");
const Error = @import("../errors.zig").Error;
const Data = @import("data.zig").Data;

/// `Stat(T)` holds per-column statistics about data.
///
/// Statistics are computed over the rows of `data.x`: minimum, maximum, sum,
/// mean, and population standard deviation for each feature.
pub fn Stat(comptime T: type) type {
    return struct {
        name: []const u8,
        data: Data(T),
        min_x: []T,
        max_x: []T,
        sum_x: []T,
        mean_x: []T,
        std_x: []T,

        const Self = @This();

        /// Allocates and computes statistics from `data`.
        pub fn from_data(
            data: Data(T),
            name: []const u8,
            allocator: std.mem.Allocator,
        ) Error!Self {
            if (data.nb_features == 0) return error.InvalidDimension;

            const n_features = data.nb_features;
            const min_x = try allocator.alloc(T, n_features);
            errdefer allocator.free(min_x);
            const max_x = try allocator.alloc(T, n_features);
            errdefer allocator.free(max_x);
            const sum_x = try allocator.alloc(T, n_features);
            errdefer allocator.free(sum_x);
            const mean_x = try allocator.alloc(T, n_features);
            errdefer allocator.free(mean_x);
            const std_x = try allocator.alloc(T, n_features);
            errdefer allocator.free(std_x);

            var self = Self{
                .name = name,
                .data = data,
                .min_x = min_x,
                .max_x = max_x,
                .sum_x = sum_x,
                .mean_x = mean_x,
                .std_x = std_x,
            };
            self.update();
            return self;
        }

        /// Recomputes all per-column statistics from `self.data.x`.
        pub fn update(self: *Self) void {
            const m = self.data.x.rows;
            const n = self.data.x.cols;
            if (m == 0 or n == 0) return;

            const mf: T = @floatFromInt(m);
            for (0..n) |j| {
                var min_val = self.data.x.get(0, j) catch unreachable;
                var max_val = min_val;
                var sum_val: T = 0.0;

                for (0..m) |i| {
                    const xval = self.data.x.get(i, j) catch unreachable;
                    if (xval < min_val) min_val = xval;
                    if (xval > max_val) max_val = xval;
                    sum_val += xval;
                }

                const mean_val = sum_val / mf;
                var sum_sq: T = 0.0;
                for (0..m) |i| {
                    const diff = (self.data.x.get(i, j) catch unreachable) - mean_val;
                    sum_sq += diff * diff;
                }

                self.min_x[j] = min_val;
                self.max_x[j] = max_val;
                self.sum_x[j] = sum_val;
                self.mean_x[j] = mean_val;
                self.std_x[j] = std.math.sqrt(sum_sq / mf);
            }
        }

        /// Releases all owned memory.
        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.min_x);
            allocator.free(self.max_x);
            allocator.free(self.sum_x);
            allocator.free(self.mean_x);
            allocator.free(self.std_x);
            self.min_x = &[_]T{};
            self.max_x = &[_]T{};
            self.sum_x = &[_]T{};
            self.mean_x = &[_]T{};
            self.std_x = &[_]T{};
        }
    };
}

test "Stat from_data computes per-column statistics" {
    const D = Data(f64);
    const xraw = &[_][]const f64{
        &[_]f64{ 1.0, 2.0, 3.0 },
        &[_]f64{ 4.0, 5.0, 6.0 },
        &[_]f64{ 7.0, 8.0, 9.0 },
    };
    var data = try D.fromRawX(std.testing.allocator, xraw);
    defer data.deinit(std.testing.allocator);

    var stat = try Stat(f64).from_data(data, "test_stat", std.testing.allocator);
    defer stat.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("test_stat", stat.name);

    // Column 0: [1, 4, 7] -> min 1, max 7, sum 12, mean 4, std = sqrt(((9+0+9)/3)) = sqrt(6)
    try std.testing.expectApproxEqAbs(1.0, stat.min_x[0], 1e-12);
    try std.testing.expectApproxEqAbs(7.0, stat.max_x[0], 1e-12);
    try std.testing.expectApproxEqAbs(12.0, stat.sum_x[0], 1e-12);
    try std.testing.expectApproxEqAbs(4.0, stat.mean_x[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(6.0), stat.std_x[0], 1e-12);

    // Column 1: [2, 5, 8] -> same std
    try std.testing.expectApproxEqAbs(std.math.sqrt(6.0), stat.std_x[1], 1e-12);

    // Column 2: [3, 6, 9] -> same std
    try std.testing.expectApproxEqAbs(std.math.sqrt(6.0), stat.std_x[2], 1e-12);
}

test "Stat update recomputes after data changes" {
    const D = Data(f64);
    const xraw = &[_][]const f64{
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 3.0, 4.0 },
    };
    var data = try D.fromRawX(std.testing.allocator, xraw);
    defer data.deinit(std.testing.allocator);

    var stat = try Stat(f64).from_data(data, "test_stat", std.testing.allocator);
    defer stat.deinit(std.testing.allocator);

    try std.testing.expectApproxEqAbs(2.0, stat.mean_x[0], 1e-12);

    // Modify data and update statistics.
    try stat.data.x.set(0, 0, 5.0);
    stat.update();

    try std.testing.expectApproxEqAbs(4.0, stat.mean_x[0], 1e-12);
    try std.testing.expectApproxEqAbs(5.0, stat.max_x[0], 1e-12);
}
