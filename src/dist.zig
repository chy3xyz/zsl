const std = @import("std");
const Error = @import("errors.zig").Error;

/// Result of a histogram computation.
/// `counts` has length `bins`; `edges` has length `bins + 1`.
/// Bin `i` covers the half-open interval `[edges[i], edges[i + 1])`,
/// except the final bin which also includes the rightmost edge.
pub const HistogramResult = struct {
    counts: []usize,
    edges: []f64,

    /// Release the allocated slices.
    pub fn deinit(self: HistogramResult, allocator: std.mem.Allocator) void {
        allocator.free(self.counts);
        allocator.free(self.edges);
    }
};

/// Compute a histogram of `data` using `bins` equal-width intervals.
/// Returns `error.InvalidDimension` if `data` is empty or `bins` is zero.
pub fn hist(data: []const f64, bins: usize, allocator: std.mem.Allocator) Error!HistogramResult {
    if (data.len == 0 or bins == 0) {
        return error.InvalidDimension;
    }

    var min = data[0];
    var max = data[0];
    for (data[1..]) |x| {
        if (x < min) min = x;
        if (x > max) max = x;
    }

    const counts = try allocator.alloc(usize, bins);
    errdefer allocator.free(counts);
    @memset(counts, 0);

    const edges = try allocator.alloc(f64, bins + 1);
    errdefer allocator.free(edges);

    edges[0] = min;
    edges[bins] = max;

    if (min == max) {
        // Degenerate range: all values fall in the first bin.
        @memset(edges[1..bins], min);
        counts[0] = data.len;
    } else {
        const width = (max - min) / @as(f64, @floatFromInt(bins));
        for (1..bins) |i| {
            edges[i] = min + width * @as(f64, @floatFromInt(i));
        }

        for (data) |x| {
            const idx: usize = if (x >= max)
                bins - 1
            else
                @intFromFloat((x - min) / width);
            counts[idx] += 1;
        }
    }

    return HistogramResult{
        .counts = counts,
        .edges = edges,
    };
}

test "hist produces expected counts and edges" {
    const allocator = std.testing.allocator;
    const data = &[_]f64{ 0.5, 1.5, 2.5, 3.5, 4.5 };

    const result = try hist(data, 2, allocator);
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), result.counts.len);
    try std.testing.expectEqual(@as(usize, 3), result.edges.len);

    try std.testing.expectEqual(@as(usize, 2), result.counts[0]);
    try std.testing.expectEqual(@as(usize, 3), result.counts[1]);

    try std.testing.expectApproxEqAbs(result.edges[0], 0.5, 1e-12);
    try std.testing.expectApproxEqAbs(result.edges[1], 2.5, 1e-12);
    try std.testing.expectApproxEqAbs(result.edges[2], 4.5, 1e-12);
}

test "hist rejects empty data and zero bins" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.InvalidDimension, hist(&[_]f64{}, 5, allocator));
    try std.testing.expectError(error.InvalidDimension, hist(&[_]f64{1.0}, 0, allocator));
}

test "hist handles constant data" {
    const allocator = std.testing.allocator;
    const data = &[_]f64{ 3.0, 3.0, 3.0 };

    const result = try hist(data, 4, allocator);
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 4), result.counts.len);
    try std.testing.expectEqual(@as(usize, 5), result.edges.len);

    try std.testing.expectEqual(@as(usize, 3), result.counts[0]);
    for (result.counts[1..]) |c| {
        try std.testing.expectEqual(@as(usize, 0), c);
    }
    for (result.edges) |e| {
        try std.testing.expectApproxEqAbs(e, 3.0, 1e-12);
    }
}
