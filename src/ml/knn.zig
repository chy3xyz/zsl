const std = @import("std");
const Error = @import("../errors.zig").Error;
const Data = @import("data.zig").Data;

/// Hash context that allows `f64` keys in a `std.HashMap`.
const F64Context = struct {
    pub fn hash(_: @This(), key: f64) u64 {
        const bits: u64 = @bitCast(key);
        return bits *% 0x9e3779b97f4a7c15;
    }

    pub fn eql(_: @This(), a: f64, b: f64) bool {
        return a == b;
    }
};

/// Weighted map keyed by class value.
const WeightsMap = std.HashMap(f64, f64, F64Context, std.hash_map.default_max_load_percentage);

/// A single training sample stored for nearest-neighbor search.
pub const Neighbor = struct {
    point: []const f64,
    class: f64,
    distance: f64,
};

/// K-Nearest Neighbors classifier for `f64` data.
pub const KNN = struct {
    name: []const u8,
    data: *Data(f64),
    weights: WeightsMap,
    neighbors: []Neighbor,
    trained: bool,

    const Self = @This();

    /// Creates a new KNN model backed by `data`.
    ///
    /// The model copies every sample row into `neighbors[i].point` and assigns a
    /// default weight of `1.0` to each distinct class in `data.y`.
    pub fn init(data: *Data(f64), name: []const u8, allocator: std.mem.Allocator) Error!Self {
        if (data.x.data.len == 0) return error.InvalidDimension;
        if (data.y.len == 0) return error.InvalidDimension;

        const neighbors = try allocator.alloc(Neighbor, data.nb_samples);
        errdefer allocator.free(neighbors);

        var i: usize = 0;
        errdefer {
            for (0..i) |j| {
                allocator.free(@constCast(neighbors[j].point));
            }
        }

        while (i < data.nb_samples) : (i += 1) {
            const point = try allocator.alloc(f64, data.nb_features);
            for (0..data.nb_features) |j| {
                point[j] = try data.x.get(i, j);
            }
            neighbors[i] = .{
                .point = point,
                .class = data.y[i],
                .distance = 0.0,
            };
        }

        var weights = WeightsMap.init(allocator);
        errdefer weights.deinit();

        for (data.y) |class| {
            try weights.put(class, 1.0);
        }

        return .{
            .name = name,
            .data = data,
            .weights = weights,
            .neighbors = neighbors,
            .trained = false,
        };
    }

    /// Releases all memory owned by the model.
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        for (self.neighbors) |neighbor| {
            allocator.free(@constCast(neighbor.point));
        }
        allocator.free(self.neighbors);
        self.weights.deinit();
        self.neighbors = &[_]Neighbor{};
        self.trained = false;
    }

    /// Replaces the per-class weights.
    ///
    /// Every weight key must be present in `data.y` and every weight value must
    /// be non-zero. Missing classes receive a default weight of `1.0`.
    pub fn set_weights(self: *Self, weights: WeightsMap) Error!void {
        var new_weights = WeightsMap.init(self.weights.allocator);
        errdefer new_weights.deinit();

        var iter = weights.iterator();
        while (iter.next()) |entry| {
            const class = entry.key_ptr.*;
            const value = entry.value_ptr.*;
            if (!self.containsClass(class)) return error.ShapeMismatch;
            if (value == 0.0) return error.DivisionByZero;
            try new_weights.put(class, value);
        }

        for (self.data.y) |class| {
            const gop = try new_weights.getOrPut(class);
            if (!gop.found_existing) {
                gop.value_ptr.* = 1.0;
            }
        }

        self.weights.deinit();
        self.weights = new_weights;
    }

    fn containsClass(self: Self, class: f64) bool {
        for (self.data.y) |c| {
            if (c == class) return true;
        }
        return false;
    }

    /// Rebuilds the default weights and marks the model as trained.
    pub fn train(self: *Self) Error!void {
        self.weights.clearRetainingCapacity();
        for (self.data.y) |class| {
            try self.weights.put(class, 1.0);
        }
        self.trained = true;
    }

    /// Configuration for `predict`.
    pub const PredictConfig = struct {
        k: usize,
        to_pred: []const f64,
        max_iter: i32 = 100,
    };

    /// Predicts the class of `to_pred` using the `k` nearest neighbors.
    ///
    /// Ties in the class vote are broken by decreasing `k` until the tie is
    /// resolved or `k` reaches one.
    pub fn predict(self: *Self, config: PredictConfig) Error!f64 {
        if (config.k == 0) return error.InvalidDimension;
        if (config.to_pred.len == 0) return error.InvalidDimension;
        if (config.to_pred.len != self.data.nb_features) return error.ShapeMismatch;

        for (self.neighbors) |*neighbor| {
            var dist: f64 = 0.0;
            for (0..self.data.nb_features) |j| {
                const diff = config.to_pred[j] - neighbor.point[j];
                dist += diff * diff;
            }
            const weight = self.weights.get(neighbor.class) orelse 1.0;
            neighbor.distance = std.math.sqrt(dist) / weight;
        }

        std.mem.sort(Neighbor, self.neighbors, {}, struct {
            fn lessThan(_: void, a: Neighbor, b: Neighbor) bool {
                return a.distance < b.distance;
            }
        }.lessThan);

        var new_k = config.k;
        if (new_k > self.neighbors.len) {
            new_k = self.neighbors.len;
        }

        var most_shown: f64 = self.data.y[0];
        var iter_number: i32 = 0;

        while (true) {
            if (config.max_iter != 0 and iter_number >= config.max_iter) {
                break;
            }

            var counts = WeightsMap.init(self.weights.allocator);
            defer counts.deinit();

            for (self.neighbors[0..new_k]) |neighbor| {
                const gop = try counts.getOrPut(neighbor.class);
                if (!gop.found_existing) {
                    gop.value_ptr.* = 0.0;
                }
                gop.value_ptr.* += 1.0;
            }

            var most_times: f64 = 0.0;
            var it = counts.iterator();
            most_shown = self.data.y[0];
            while (it.next()) |entry| {
                if (entry.value_ptr.* > most_times) {
                    most_times = entry.value_ptr.*;
                    most_shown = entry.key_ptr.*;
                }
            }

            var tied = false;
            var tie_it = counts.iterator();
            while (tie_it.next()) |entry| {
                if (entry.key_ptr.* != most_shown and entry.value_ptr.* == most_times) {
                    tied = true;
                    break;
                }
            }

            if (!tied) break;
            if (new_k == 1) break;

            new_k -= 1;

            if (config.max_iter != 0) {
                iter_number += 1;
            }
        }

        return most_shown;
    }
};

test "KNN predicts closest class" {
    const allocator = std.testing.allocator;
    const D = Data(f64);

    const xraw = &[_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 10.0, 10.0 },
    };
    const yraw = &[_]f64{ 0.0, 1.0 };

    var data = try D.fromRawXy(allocator, xraw, yraw);
    defer data.deinit(allocator);

    var knn = try KNN.init(&data, "test_knn", allocator);
    defer knn.deinit(allocator);

    try knn.train();

    const prediction = try knn.predict(.{ .k = 1, .to_pred = &[_]f64{ 9.0, 9.0 } });
    try std.testing.expectApproxEqAbs(1.0, prediction, 1e-12);
}
