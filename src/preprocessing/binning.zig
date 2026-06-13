const std = @import("std");
const Error = @import("../errors.zig").Error;

fn findBin(value: f64, edges: []const f64) usize {
    const n_bins = edges.len - 1;
    for (0..n_bins) |i| {
        if (value <= edges[i + 1]) return i;
    }
    return n_bins - 1;
}

fn assignBins(
    values: []const f64,
    edges: []const f64,
    labels: ?[]const []const u8,
    allocator: std.mem.Allocator,
) Error![]const []const u8 {
    const n_bins = edges.len - 1;
    if (labels) |ls| {
        if (ls.len != n_bins) return error.ShapeMismatch;
    }

    var result = try allocator.alloc([]const u8, values.len);
    errdefer {
        for (result) |s| allocator.free(s);
        allocator.free(result);
    }

    for (values, 0..) |v, i| {
        const idx = findBin(v, edges);
        result[i] = if (labels) |ls|
            try allocator.dupe(u8, ls[idx])
        else
            try std.fmt.allocPrint(allocator, "bin_{d}", .{idx});
    }
    return result;
}

/// Bins `values` into `n_bins` equal-width intervals.
/// When `labels` is null, labels are auto-generated as `bin_0`, `bin_1`, ...
/// The returned slice and each returned label string are allocated and must be
/// freed by the caller.
pub fn cut(
    values: []const f64,
    n_bins: usize,
    labels: ?[]const []const u8,
    allocator: std.mem.Allocator,
) Error![]const []const u8 {
    if (values.len == 0) return error.InvalidDimension;
    if (n_bins < 2) return error.InvalidDimension;

    var min_val = values[0];
    var max_val = values[0];
    for (values) |v| {
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    const bin_width = (max_val - min_val) / @as(f64, @floatFromInt(n_bins));
    var edges = try allocator.alloc(f64, n_bins + 1);
    defer allocator.free(edges);
    for (0..n_bins + 1) |i| {
        edges[i] = min_val + @as(f64, @floatFromInt(i)) * bin_width;
    }

    return try assignBins(values, edges, labels, allocator);
}

/// Bins `values` into `n_bins` quantile-based intervals.
/// When `labels` is null, labels are auto-generated as `bin_0`, `bin_1`, ...
/// The returned slice and each returned label string are allocated and must be
/// freed by the caller.
pub fn qcut(
    values: []const f64,
    n_bins: usize,
    labels: ?[]const []const u8,
    allocator: std.mem.Allocator,
) Error![]const []const u8 {
    if (values.len == 0) return error.InvalidDimension;
    if (n_bins < 2) return error.InvalidDimension;

    const sorted = try allocator.alloc(f64, values.len);
    defer allocator.free(sorted);
    @memcpy(sorted, values);
    std.mem.sort(f64, sorted, {}, std.sort.asc(f64));

    var edges = try allocator.alloc(f64, n_bins + 1);
    defer allocator.free(edges);
    edges[0] = sorted[0];
    edges[n_bins] = sorted[sorted.len - 1];
    for (1..n_bins) |i| {
        const q = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n_bins));
        const idx = @as(usize, @intFromFloat(q * @as(f64, @floatFromInt(sorted.len - 1))));
        edges[i] = sorted[idx];
    }

    return try assignBins(values, edges, labels, allocator);
}

test "cut with auto labels" {
    const values = &[_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
    const labels = try cut(values, 3, null, std.testing.allocator);
    defer {
        for (labels) |s| std.testing.allocator.free(s);
        std.testing.allocator.free(labels);
    }
    try std.testing.expectEqualStrings("bin_0", labels[0]);
    try std.testing.expectEqualStrings("bin_0", labels[1]);
    try std.testing.expectEqualStrings("bin_1", labels[2]);
    try std.testing.expectEqualStrings("bin_1", labels[3]);
    try std.testing.expectEqualStrings("bin_2", labels[4]);
    try std.testing.expectEqualStrings("bin_2", labels[5]);
}

test "cut with provided labels" {
    const values = &[_]f64{ 0.0, 5.0, 10.0 };
    const custom = &[_][]const u8{ "low", "mid", "high" };
    const labels = try cut(values, 3, custom, std.testing.allocator);
    defer {
        for (labels) |s| std.testing.allocator.free(s);
        std.testing.allocator.free(labels);
    }
    try std.testing.expectEqualStrings("low", labels[0]);
    try std.testing.expectEqualStrings("mid", labels[1]);
    try std.testing.expectEqualStrings("high", labels[2]);
}

test "qcut with auto labels" {
    const values = &[_]f64{ 0.0, 10.0, 20.0, 30.0, 40.0, 50.0 };
    const labels = try qcut(values, 3, null, std.testing.allocator);
    defer {
        for (labels) |s| std.testing.allocator.free(s);
        std.testing.allocator.free(labels);
    }
    try std.testing.expectEqualStrings("bin_0", labels[0]);
    try std.testing.expectEqualStrings("bin_0", labels[1]);
    try std.testing.expectEqualStrings("bin_1", labels[2]);
    try std.testing.expectEqualStrings("bin_1", labels[3]);
    try std.testing.expectEqualStrings("bin_2", labels[4]);
    try std.testing.expectEqualStrings("bin_2", labels[5]);
}
