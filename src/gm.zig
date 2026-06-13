const std = @import("std");
const Error = @import("errors.zig").Error;

pub const point = @import("gm/point.zig");
pub const segment = @import("gm/segment.zig");

pub const Point = point.Point;
pub const Segment = segment.Segment;

test {
    _ = point;
    _ = segment;
}

/// Minimum distance between coordinates along any dimension.
const xdelzero: f64 = 1e-10;

/// Entry stored inside a bin.
pub const BinEntry = struct {
    id: usize,
    x: []const f64,
    extra: ?*anyopaque,
};

/// One bin holding a list of entries.
pub const Bin = struct {
    index: usize,
    entries: std.ArrayListUnmanaged(BinEntry),
};

/// Set of bins used for fast spatial searches.
pub const Bins = struct {
    ndim: usize,
    xmin: []const f64,
    xmax: []const f64,
    xdel: []f64,
    size: []f64,
    ndiv: []usize,
    all: []?Bin,
    tmp: []usize,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Initialise a new `Bins` structure.
    ///
    /// `xmin` and `xmax` define the bounding box and `ndiv` the number of
    /// divisions along each dimension. Only 2D and 3D spaces are supported.
    pub fn init(xmin: []const f64, xmax: []const f64, ndiv: []const usize, allocator: std.mem.Allocator) Error!Self {
        if (xmin.len != xmax.len or xmin.len != ndiv.len) return error.ShapeMismatch;
        const ndim = xmin.len;
        if (ndim < 2 or ndim > 3) return error.InvalidDimension;

        const xmin_copy = try allocator.dupe(f64, xmin);
        errdefer allocator.free(xmin_copy);
        const xmax_copy = try allocator.dupe(f64, xmax);
        errdefer allocator.free(xmax_copy);
        const ndiv_copy = try allocator.alloc(usize, ndim);
        errdefer allocator.free(ndiv_copy);
        const xdel = try allocator.alloc(f64, ndim);
        errdefer allocator.free(xdel);
        const size = try allocator.alloc(f64, ndim);
        errdefer allocator.free(size);

        var nbins: u128 = 1;
        for (0..ndim) |k| {
            if (ndiv[k] == 0) return error.InvalidDimension;
            ndiv_copy[k] = ndiv[k];
            xdel[k] = xmax[k] - xmin[k];
            if (xdel[k] < xdelzero) return error.InvalidDimension;
            size[k] = xdel[k] / @as(f64, @floatFromInt(ndiv[k]));
            nbins *= ndiv[k];
        }
        if (nbins > std.math.maxInt(usize)) return error.InvalidDimension;

        const all = try allocator.alloc(?Bin, @intCast(nbins));
        errdefer allocator.free(all);
        for (all, 0..) |*slot, i| {
            slot.* = .{ .index = i, .entries = .empty };
        }

        const tmp = try allocator.alloc(usize, ndim);
        errdefer allocator.free(tmp);

        return .{
            .ndim = ndim,
            .xmin = xmin_copy,
            .xmax = xmax_copy,
            .xdel = xdel,
            .size = size,
            .ndiv = ndiv_copy,
            .all = all,
            .tmp = tmp,
            .allocator = allocator,
        };
    }

    /// Release all resources held by the bins.
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.clear();
        for (self.all) |*maybe_bin| {
            if (maybe_bin.*) |*bin| {
                bin.entries.deinit(allocator);
            }
        }
        allocator.free(self.all);
        allocator.free(self.tmp);
        allocator.free(self.xdel);
        allocator.free(self.size);
        allocator.free(self.ndiv);
        allocator.free(self.xmin);
        allocator.free(self.xmax);
    }

    /// Add a new entry at coordinate `x` with identifier `id` and optional
    /// extra pointer.
    pub fn append(self: *Self, x: []const f64, id: usize, extra: ?*anyopaque) Error!void {
        if (x.len != self.ndim) return error.ShapeMismatch;
        const idx = self.calc_index(x);
        if (idx < 0) return error.IndexOutOfBounds;

        const xcopy = try self.allocator.dupe(f64, x);
        errdefer self.allocator.free(xcopy);

        const bin = self.find_bin_by_index(idx) orelse return error.IndexOutOfBounds;
        try bin.entries.append(self.allocator, .{
            .id = id,
            .x = xcopy,
            .extra = extra,
        });
    }

    /// Remove all entries from every bin.
    pub fn clear(self: *Self) void {
        for (self.all) |*maybe_bin| {
            if (maybe_bin.*) |*bin| {
                for (bin.entries.items) |entry| {
                    self.allocator.free(entry.x);
                }
                bin.entries.clearAndFree(self.allocator);
            }
        }
    }

    /// Compute the flat bin index containing coordinate `x`, or `-1` if `x` is
    /// outside the bounding box.
    pub fn calc_index(self: *Self, x: []const f64) isize {
        if (x.len != self.ndim) return -1;
        for (0..self.ndim) |k| {
            if (x[k] < self.xmin[k] or x[k] > self.xmax[k]) return -1;
            var idx_dim: usize = @intFromFloat((x[k] - self.xmin[k]) / self.size[k]);
            if (idx_dim == self.ndiv[k]) idx_dim -= 1;
            self.tmp[k] = idx_dim;
        }
        var idx: isize = @intCast(self.tmp[0] + self.tmp[1] * self.ndiv[0]);
        if (self.ndim > 2) {
            idx += @intCast(self.tmp[2] * self.ndiv[0] * self.ndiv[1]);
        }
        return idx;
    }

    /// Return the bin associated with flat index `idx`.
    pub fn find_bin_by_index(self: *Self, idx: isize) ?*Bin {
        if (idx < 0 or idx >= self.all.len) return null;
        return &self.all[@intCast(idx)].?;
    }

    /// Return the first entry within Euclidean distance `tol` of `x`.
    pub fn find(self: *Self, x: []const f64, tol: f64) Error!?BinEntry {
        if (x.len != self.ndim) return error.ShapeMismatch;
        const idx = self.calc_index(x);
        if (idx < 0) return null;
        const bin = self.find_bin_by_index(idx) orelse return null;
        const tol_sq = tol * tol;
        for (bin.entries.items) |entry| {
            if (dist_sq(self.ndim, entry.x, x) <= tol_sq) {
                return entry;
            }
        }
        return null;
    }

    /// Return all entries within Euclidean distance `tol` of `x`.
    pub fn find_all(self: *Self, x: []const f64, tol: f64, allocator: std.mem.Allocator) Error![]BinEntry {
        if (x.len != self.ndim) return error.ShapeMismatch;
        const idx = self.calc_index(x);
        if (idx < 0) {
            return try allocator.alloc(BinEntry, 0);
        }
        const bin = self.find_bin_by_index(idx) orelse return try allocator.alloc(BinEntry, 0);

        var result: std.ArrayList(BinEntry) = .empty;
        errdefer result.deinit(allocator);

        const tol_sq = tol * tol;
        for (bin.entries.items) |entry| {
            if (dist_sq(self.ndim, entry.x, x) <= tol_sq) {
                try result.append(allocator, entry);
            }
        }
        return result.toOwnedSlice(allocator);
    }
};

fn dist_sq(ndim: usize, a: []const f64, b: []const f64) f64 {
    var d: f64 = 0.0;
    for (0..ndim) |k| {
        const dk = a[k] - b[k];
        d += dk * dk;
    }
    return d;
}

test "2D bins append, find and find_all" {
    const allocator = std.testing.allocator;
    var bins = try Bins.init(
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 10.0, 10.0 },
        &[_]usize{ 2, 2 },
        allocator,
    );
    defer bins.deinit(allocator);

    try bins.append(&[_]f64{ 1.0, 1.0 }, 10, null);
    try bins.append(&[_]f64{ 1.5, 1.5 }, 20, null);
    try bins.append(&[_]f64{ 6.0, 6.0 }, 30, null);

    const found = try bins.find(&[_]f64{ 1.1, 1.1 }, 0.5);
    try std.testing.expect(found != null);
    try std.testing.expectEqual(@as(usize, 10), found.?.id);

    const all = try bins.find_all(&[_]f64{ 1.1, 1.1 }, 1.0, allocator);
    defer allocator.free(all);
    try std.testing.expectEqual(@as(usize, 2), all.len);
    try std.testing.expectEqual(@as(usize, 10), all[0].id);
    try std.testing.expectEqual(@as(usize, 20), all[1].id);
}

test "out-of-range coordinates return error or null" {
    const allocator = std.testing.allocator;
    var bins = try Bins.init(
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 10.0, 10.0 },
        &[_]usize{ 2, 2 },
        allocator,
    );
    defer bins.deinit(allocator);

    try std.testing.expectError(error.IndexOutOfBounds, bins.append(&[_]f64{ -1.0, 5.0 }, 1, null));
    try std.testing.expectEqual(@as(?BinEntry, null), try bins.find(&[_]f64{ 11.0, 5.0 }, 0.5));

    const all = try bins.find_all(&[_]f64{ 5.0, 11.0 }, 0.5, allocator);
    defer allocator.free(all);
    try std.testing.expectEqual(@as(usize, 0), all.len);
}

test "clear removes all entries" {
    const allocator = std.testing.allocator;
    var bins = try Bins.init(
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 10.0, 10.0 },
        &[_]usize{ 2, 2 },
        allocator,
    );
    defer bins.deinit(allocator);

    try bins.append(&[_]f64{ 1.0, 1.0 }, 1, null);
    try bins.append(&[_]f64{ 2.0, 2.0 }, 2, null);
    bins.clear();

    try std.testing.expectEqual(@as(?BinEntry, null), try bins.find(&[_]f64{ 1.0, 1.0 }, 0.1));
}
