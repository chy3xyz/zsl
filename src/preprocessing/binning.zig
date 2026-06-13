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

fn makeAutoLabels(allocator: std.mem.Allocator, n_bins: usize, comptime fmt: []const u8, offset: usize) Error![]const []const u8 {
    var result = try allocator.alloc([]const u8, n_bins);
    errdefer {
        for (result) |s| allocator.free(s);
        allocator.free(result);
    }
    for (0..n_bins) |i| {
        result[i] = try std.fmt.allocPrint(allocator, fmt, .{i + offset});
    }
    return result;
}

/// Bins `values` into `n_bins` quantile-based intervals.
/// When `labels` is null, labels are auto-generated as `Q1`, `Q2`, ...
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

    const auto_labels = if (labels == null)
        try makeAutoLabels(allocator, n_bins, "Q{d}", 1)
    else
        null;
    defer if (auto_labels) |ls| {
        for (ls) |s| allocator.free(s);
        allocator.free(ls);
    };

    return try assignBins(values, edges, labels orelse auto_labels.?, allocator);
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

/// Returns the bin index for each value given monotonically increasing bin
/// edges. Values below the first edge are mapped to 0; values above the last
/// edge are mapped to `bins.len - 2` (the last valid interval). Values exactly
/// equal to the last edge are mapped to `bins.len - 1`.
/// Intervals are treated as `[bins[i], bins[i+1])`.
pub fn digitize(allocator: std.mem.Allocator, values: []const f64, bins: []const f64) Error![]usize {
    if (bins.len < 2) return error.InvalidDimension;

    var result = try allocator.alloc(usize, values.len);
    errdefer allocator.free(result);

    for (values, 0..) |v, i| {
        result[i] = digitizeValue(v, bins);
    }
    return result;
}

fn digitizeValue(value: f64, bins: []const f64) usize {
    if (value < bins[0]) return 0;
    if (value > bins[bins.len - 1]) return bins.len - 2;
    if (value == bins[bins.len - 1]) return bins.len - 1;

    var lo: usize = 0;
    var hi: usize = bins.len - 1;
    while (lo + 1 < hi) {
        const mid = lo + (hi - lo) / 2;
        if (value < bins[mid]) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return lo;
}

test "qcut with auto labels" {
    const values = &[_]f64{ 0.0, 10.0, 20.0, 30.0, 40.0, 50.0 };
    const labels = try qcut(values, 3, null, std.testing.allocator);
    defer {
        for (labels) |s| std.testing.allocator.free(s);
        std.testing.allocator.free(labels);
    }
    try std.testing.expectEqualStrings("Q1", labels[0]);
    try std.testing.expectEqualStrings("Q1", labels[1]);
    try std.testing.expectEqualStrings("Q2", labels[2]);
    try std.testing.expectEqualStrings("Q2", labels[3]);
    try std.testing.expectEqualStrings("Q3", labels[4]);
    try std.testing.expectEqualStrings("Q3", labels[5]);
}

test "digitize basic intervals" {
    const values = &[_]f64{ 0.5, 1.5, 2.5, 3.5 };
    const bins = &[_]f64{ 1.0, 2.0, 3.0 };
    const result = try digitize(std.testing.allocator, values, bins);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqual(0, result[0]); // 0.5 < 1.0 -> clamped to first bin
    try std.testing.expectEqual(0, result[1]); // [1.0, 2.0)
    try std.testing.expectEqual(1, result[2]); // [2.0, 3.0)
    try std.testing.expectEqual(1, result[3]); // 3.5 >= 3.0 -> last bin
}

test "digitize edge values" {
    const bins = &[_]f64{ 0.0, 1.0, 2.0 };
    const values = &[_]f64{ -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0 };
    const result = try digitize(std.testing.allocator, values, bins);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqual(0, result[0]); // below
    try std.testing.expectEqual(0, result[1]); // left edge -> first bin
    try std.testing.expectEqual(0, result[2]); // first bin
    try std.testing.expectEqual(1, result[3]); // right edge of first bin -> second bin
    try std.testing.expectEqual(1, result[4]); // second bin
    try std.testing.expectEqual(2, result[5]); // right edge -> bins.len - 1
    try std.testing.expectEqual(1, result[6]); // above -> last valid interval
}

test "digitize rejects single edge" {
    const values = &[_]f64{1.0};
    const bins = &[_]f64{1.0};
    try std.testing.expectError(error.InvalidDimension, digitize(std.testing.allocator, values, bins));
}
