const std = @import("std");
const Data = @import("data.zig").Data;
const Stat = @import("workspace.zig").Stat;
const DecisionTree = @import("decision_tree.zig").DecisionTree;
const Error = @import("../errors.zig").Error;

/// A Random Forest classifier.
///
/// Trains an ensemble of decision trees on bootstrap samples of the training
/// data and aggregates predictions by majority vote.
pub const RandomForest = struct {
    name: []const u8,
    data: *Data(f64),
    stat: *Stat(f64),
    n_estimators: usize,
    max_features: usize,
    bootstrap: bool,
    trained: bool,
    trees: std.ArrayList(*DecisionTree),
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Creates a new untrained random forest.
    pub fn init(data: *Data(f64), name: []const u8, allocator: std.mem.Allocator) Error!Self {
        if (data.y.len == 0) return error.InvalidDimension;

        const stat = try allocator.create(Stat(f64));
        errdefer allocator.destroy(stat);
        stat.* = try Stat(f64).from_data(data.*, name, allocator);
        errdefer stat.deinit(allocator);
        stat.update();

        const max_features = if (data.nb_features > 0)
            @as(usize, @intFromFloat(@sqrt(@as(f64, @floatFromInt(data.nb_features)))))
        else
            data.nb_features;

        return .{
            .name = name,
            .data = data,
            .stat = stat,
            .n_estimators = 100,
            .max_features = max_features,
            .bootstrap = true,
            .trained = false,
            .trees = .empty,
            .allocator = allocator,
        };
    }

    /// Releases all owned memory.
    pub fn deinit(self: *Self) void {
        for (self.trees.items) |tree| {
            tree.deinit();
            self.allocator.destroy(tree);
        }
        self.trees.deinit(self.allocator);
        self.stat.deinit(self.allocator);
        self.allocator.destroy(self.stat);
        self.* = undefined;
    }

    /// Sets the number of trees in the forest.
    pub fn set_n_estimators(self: *Self, n: usize) Error!void {
        if (n == 0) return error.InvalidDimension;
        self.n_estimators = n;
        self.trained = false;
    }

    /// Sets whether to use bootstrap sampling when building trees.
    pub fn set_bootstrap(self: *Self, b: bool) void {
        self.bootstrap = b;
        self.trained = false;
    }

    /// Trains the ensemble.
    pub fn train(self: *Self) Error!void {
        if (self.data.nb_samples == 0) return;

        // Clear any previously trained trees.
        for (self.trees.items) |tree| {
            tree.deinit();
            self.allocator.destroy(tree);
        }
        self.trees.clearRetainingCapacity();

        for (0..self.n_estimators) |e| {
            const indices = if (self.bootstrap)
                try bootstrap_sample(self.allocator, self.data.nb_samples, @intCast(e))
            else
                try identity_sample(self.allocator, self.data.nb_samples);
            defer self.allocator.free(indices);

            var subset = try make_bootstrap_subset(self.allocator, self.data, indices);
            defer subset.deinit(self.allocator);

            const tree = try self.allocator.create(DecisionTree);
            errdefer self.allocator.destroy(tree);
            tree.* = try DecisionTree.init(.{ .criterion = .gini, .max_depth = 10 }, self.allocator);
            errdefer tree.deinit();

            try tree.fit(&subset);

            try self.trees.append(self.allocator, tree);
        }

        self.trained = true;
    }

    /// Predicts the class label for a single sample using majority vote.
    pub fn predict(self: *Self, x: []const f64) Error!f64 {
        if (!self.trained) return error.NotFitted;
        if (x.len != self.data.nb_features) return error.ShapeMismatch;
        if (self.trees.items.len == 0) return error.NotFitted;

        const predictions = try self.allocator.alloc(f64, self.trees.items.len);
        defer self.allocator.free(predictions);

        for (self.trees.items, 0..) |tree, i| {
            predictions[i] = try tree.predict(x);
        }

        return majority_class(predictions);
    }

    /// Returns the fraction of trees that predict class `1.0`.
    pub fn predict_proba(self: *Self, x: []const f64) Error!f64 {
        if (!self.trained) return error.NotFitted;
        if (x.len != self.data.nb_features) return error.ShapeMismatch;
        if (self.trees.items.len == 0) return error.NotFitted;

        var votes_1: usize = 0;
        for (self.trees.items) |tree| {
            const pred = try tree.predict(x);
            if (pred == 1.0) votes_1 += 1;
        }

        return @as(f64, @floatFromInt(votes_1)) / @as(f64, @floatFromInt(self.trees.items.len));
    }

    /// Returns a placeholder feature-importance vector of zeros.
    pub fn feature_importance(self: *Self, allocator: std.mem.Allocator) Error![]f64 {
        if (!self.trained) return error.NotFitted;

        const importance = try allocator.alloc(f64, self.data.nb_features);
        @memset(importance, 0.0);
        return importance;
    }
};

fn bootstrap_sample(allocator: std.mem.Allocator, n: usize, seed: u32) Error![]usize {
    const indices = try allocator.alloc(usize, n);
    errdefer allocator.free(indices);
    var s: u32 = seed;
    for (0..n) |i| {
        s = (s *% 1103515245 +% 12345) & 0x7fffffff;
        indices[i] = @intCast(s % @as(u32, @intCast(n)));
    }
    return indices;
}

fn identity_sample(allocator: std.mem.Allocator, n: usize) Error![]usize {
    const indices = try allocator.alloc(usize, n);
    errdefer allocator.free(indices);
    for (0..n) |i| {
        indices[i] = i;
    }
    return indices;
}

fn make_bootstrap_subset(
    allocator: std.mem.Allocator,
    data: *Data(f64),
    indices: []const usize,
) Error!Data(f64) {
    const n = indices.len;
    const nb_features = data.nb_features;

    const xrows = try allocator.alloc([]f64, n);
    errdefer allocator.free(xrows);

    var allocated_rows: usize = 0;
    errdefer {
        for (0..allocated_rows) |j| allocator.free(xrows[j]);
    }

    for (0..n) |i| {
        xrows[i] = try allocator.alloc(f64, nb_features);
        allocated_rows += 1;
        const src_idx = indices[i];
        for (0..nb_features) |j| {
            xrows[i][j] = try data.x.get(src_idx, j);
        }
    }

    const yraw = try allocator.alloc(f64, n);
    errdefer allocator.free(yraw);
    for (0..n) |i| {
        yraw[i] = data.y[indices[i]];
    }

    var subset = try Data(f64).fromRawXy(allocator, xrows, yraw);
    errdefer subset.deinit(allocator);

    for (0..n) |i| allocator.free(xrows[i]);
    allocator.free(xrows);
    allocator.free(yraw);

    return subset;
}

fn majority_class(labels: []const f64) f64 {
    if (labels.len == 0) return 0.0;
    var best_label = labels[0];
    var best_count: usize = 0;
    for (labels, 0..) |label, i| {
        if (already_seen(labels[0..i], label)) continue;
        const count = count_label(labels, label);
        if (count > best_count) {
            best_count = count;
            best_label = label;
        }
    }
    return best_label;
}

fn already_seen(labels: []const f64, label: f64) bool {
    for (labels) |l| {
        if (l == label) return true;
    }
    return false;
}

fn count_label(labels: []const f64, label: f64) usize {
    var count: usize = 0;
    for (labels) |l| {
        if (l == label) count += 1;
    }
    return count;
}

test {
    const allocator = std.testing.allocator;

    const xraw = &[_][]const f64{
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 2.0, 1.0 },
        &[_]f64{ 2.0, 2.0 },
    };
    const yraw = &[_]f64{ 0.0, 0.0, 1.0, 1.0 };

    var data = try Data(f64).fromRawXy(allocator, xraw, yraw);
    defer data.deinit(allocator);

    var rf = try RandomForest.init(&data, "test_rf", allocator);
    defer rf.deinit();

    try std.testing.expectError(error.NotFitted, rf.predict(&[_]f64{ 1.0, 1.0 }));

    try rf.set_n_estimators(5);
    try rf.train();

    const pred = try rf.predict(&[_]f64{ 1.0, 1.0 });
    try std.testing.expectApproxEqAbs(0.0, pred, 1e-12);

    const proba = try rf.predict_proba(&[_]f64{ 2.0, 2.0 });
    try std.testing.expect(proba >= 0.0 and proba <= 1.0);

    const importance = try rf.feature_importance(allocator);
    defer allocator.free(importance);
    try std.testing.expectEqual(data.nb_features, importance.len);
    for (importance) |v| try std.testing.expectApproxEqAbs(0.0, v, 1e-12);
}
