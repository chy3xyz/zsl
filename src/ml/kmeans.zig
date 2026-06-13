const std = @import("std");
const la = @import("../la.zig");
const gm = @import("../gm.zig");
const Error = @import("../errors.zig").Error;
const Data = @import("data.zig").Data;
const Stat = @import("workspace.zig").Stat;

/// `Kmeans` implements the K-means clustering algorithm for `f64` data.
///
/// The model keeps a reference to the input `Data(f64)`, per-column statistics,
/// class assignments, centroids, and a 2D spatial binning structure for fast
/// lookups (using the first two features).
pub const Kmeans = struct {
    name: []const u8,
    data: *Data(f64),
    stat: *Stat(f64),
    nb_classes: usize,
    classes: []usize,
    centroids: la.Matrix(f64),
    nb_members: []usize,
    bins: gm.Bins,
    nb_iter: usize,

    /// Workspace used to store the previous centroids during training.
    prev_centroids: la.Matrix(f64),

    const Self = @This();

    /// Creates a new K-means model.
    ///
    /// `nb_classes` is the desired number of clusters. The caller must later
    /// provide initial centroids via `set_centroids` before calling `train`.
    /// Bins are built from the first two features of the data.
    pub fn init(
        data: *Data(f64),
        nb_classes: usize,
        name: []const u8,
        allocator: std.mem.Allocator,
    ) Error!Self {
        if (nb_classes == 0 or data.nb_samples == 0) return error.InvalidDimension;
        if (data.nb_features < 2) return error.InvalidDimension;

        // Per-column statistics.
        const stat_ptr = try allocator.create(Stat(f64));
        errdefer allocator.destroy(stat_ptr);
        stat_ptr.* = try Stat(f64).from_data(data.*, "stat", allocator);
        errdefer stat_ptr.deinit(allocator);

        // Class assignments and member counts.
        const classes = try allocator.alloc(usize, data.nb_samples);
        errdefer allocator.free(classes);
        @memset(classes, 0);

        const nb_members = try allocator.alloc(usize, nb_classes);
        errdefer allocator.free(nb_members);
        @memset(nb_members, 0);

        // Centroid matrix and training workspace.
        var centroids = try la.Matrix(f64).init(allocator, nb_classes, data.nb_features);
        errdefer centroids.deinit(allocator);

        var prev_centroids = try la.Matrix(f64).init(allocator, nb_classes, data.nb_features);
        errdefer prev_centroids.deinit(allocator);

        // Spatial bins covering the first two dimensions.
        const ndiv = &[_]usize{ 10, 10 };
        var bins = try gm.Bins.init(
            stat_ptr.min_x[0..2],
            stat_ptr.max_x[0..2],
            ndiv,
            allocator,
        );
        errdefer bins.deinit(allocator);

        var self = Self{
            .name = name,
            .data = data,
            .stat = stat_ptr,
            .nb_classes = nb_classes,
            .classes = classes,
            .centroids = centroids,
            .nb_members = nb_members,
            .bins = bins,
            .nb_iter = 0,
            .prev_centroids = prev_centroids,
        };
        errdefer self.deinit(allocator);

        // Add all data points to the bins (2D coordinates only).
        for (0..data.nb_samples) |i| {
            const x = [2]f64{
                try data.x.get(i, 0),
                try data.x.get(i, 1),
            };
            try self.bins.append(&x, i, null);
        }

        return self;
    }

    /// Releases all memory owned by the model.
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.bins.deinit(allocator);
        self.centroids.deinit(allocator);
        self.prev_centroids.deinit(allocator);
        self.stat.deinit(allocator);
        allocator.destroy(self.stat);
        allocator.free(self.classes);
        allocator.free(self.nb_members);
        self.classes = &[_]usize{};
        self.nb_members = &[_]usize{};
    }

    /// Copies externally supplied centroids into the model.
    ///
    /// `xc` must have shape `[nb_classes][nb_features]`.
    pub fn set_centroids(self: *Self, xc: la.Matrix(f64)) Error!void {
        if (xc.rows != self.nb_classes or xc.cols != self.data.nb_features) {
            return error.ShapeMismatch;
        }
        for (0..self.nb_classes) |i| {
            for (0..self.data.nb_features) |j| {
                try self.centroids.set(i, j, try xc.get(i, j));
            }
        }
    }

    /// Assigns each sample to the nearest centroid.
    pub fn find_closest_centroids(self: *Self) Error!void {
        const n_samples = self.data.nb_samples;
        const n_features = self.data.nb_features;
        const k = self.nb_classes;

        for (0..n_samples) |i| {
            var min_dist = std.math.inf(f64);
            var best_class: usize = 0;
            for (0..k) |j| {
                var dist: f64 = 0.0;
                for (0..n_features) |f| {
                    const diff = try self.data.x.get(i, f) - try self.centroids.get(j, f);
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    best_class = j;
                }
            }
            self.classes[i] = best_class;
        }
    }

    /// Recomputes centroids as the mean of all samples assigned to each class.
    pub fn compute_centroids(self: *Self) Error!void {
        // Clear centroids and member counts.
        for (0..self.nb_classes) |k| {
            self.nb_members[k] = 0;
            for (0..self.data.nb_features) |j| {
                try self.centroids.set(k, j, 0.0);
            }
        }

        // Accumulate contributions.
        for (0..self.data.nb_samples) |i| {
            const k = self.classes[i];
            self.nb_members[k] += 1;
            for (0..self.data.nb_features) |j| {
                const val = try self.centroids.get(k, j) + try self.data.x.get(i, j);
                try self.centroids.set(k, j, val);
            }
        }

        // Scale by the number of members.
        for (0..self.nb_classes) |k| {
            const count = self.nb_members[k];
            if (count > 0) {
                const inv = 1.0 / @as(f64, @floatFromInt(count));
                for (0..self.data.nb_features) |j| {
                    const val = try self.centroids.get(k, j) * inv;
                    try self.centroids.set(k, j, val);
                }
            }
        }
    }

    /// Configuration for `train`.
    pub const TrainConfig = struct {
        epochs: usize = 100,
        tol_norm_change: f64 = 1e-9,
    };

    /// Runs the K-means assign/update loop until convergence or `epochs`.
    pub fn train(self: *Self, config: TrainConfig) Error!void {
        var iter: usize = 0;
        while (iter < config.epochs) : (iter += 1) {
            // Save current centroids for movement calculation.
            for (0..self.nb_classes) |i| {
                for (0..self.data.nb_features) |j| {
                    try self.prev_centroids.set(i, j, try self.centroids.get(i, j));
                }
            }

            try self.find_closest_centroids();
            try self.compute_centroids();

            const change = try centroidDiffNorm(self.centroids, self.prev_centroids);
            if (change < config.tol_norm_change) break;
        }
        self.nb_iter += iter;
    }
};

fn centroidDiffNorm(a: la.Matrix(f64), b: la.Matrix(f64)) Error!f64 {
    var sum: f64 = 0.0;
    for (0..a.rows) |i| {
        for (0..a.cols) |j| {
            const d = try a.get(i, j) - try b.get(i, j);
            sum += d * d;
        }
    }
    return std.math.sqrt(sum);
}

test "Kmeans clusters a separable 2D dataset" {
    const allocator = std.testing.allocator;
    const D = Data(f64);
    const M = la.Matrix(f64);

    const xraw = &[_][]const f64{
        &[_]f64{ 0.1, 0.1 },
        &[_]f64{ 0.2, 0.2 },
        &[_]f64{ -0.1, 0.1 },
        &[_]f64{ 0.1, -0.2 },
        &[_]f64{ 5.1, 5.2 },
        &[_]f64{ 5.2, 5.1 },
        &[_]f64{ 4.9, 5.0 },
        &[_]f64{ 5.0, 4.8 },
    };
    const yraw = &[_]f64{ 0, 0, 0, 0, 1, 1, 1, 1 };

    var data = try D.fromRawXy(allocator, xraw, yraw);
    defer data.deinit(allocator);

    var km = try Kmeans.init(&data, 2, "test_kmeans", allocator);
    defer km.deinit(allocator);

    var init_centroids = try M.fromRowSlice(allocator, 2, 2, &[_]f64{
        0.0, 0.0,
        5.0, 5.0,
    });
    defer init_centroids.deinit(allocator);

    try km.set_centroids(init_centroids);
    try km.train(.{ .epochs = 100, .tol_norm_change = 1e-9 });

    var counts = [_]usize{ 0, 0 };
    for (km.classes) |c| {
        counts[c] += 1;
    }
    try std.testing.expectEqual(@as(usize, 4), counts[0]);
    try std.testing.expectEqual(@as(usize, 4), counts[1]);

    // The centroid of class 0 should remain near the origin, class 1 near (5,5).
    try std.testing.expectApproxEqAbs(0.0, try km.centroids.get(0, 0), 0.2);
    try std.testing.expectApproxEqAbs(0.0, try km.centroids.get(0, 1), 0.2);
    try std.testing.expectApproxEqAbs(5.0, try km.centroids.get(1, 0), 0.2);
    try std.testing.expectApproxEqAbs(5.0, try km.centroids.get(1, 1), 0.2);
}
