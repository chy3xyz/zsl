const std = @import("std");
const Data = @import("data.zig").Data;
const Error = @import("../errors.zig").Error;

/// Splitting criterion used when building the tree.
pub const Criterion = enum {
    gini,
    entropy,
};

/// Configuration for `DecisionTree` training.
pub const TreeConfig = struct {
    max_depth: usize = 10,
    min_samples_split: usize = 2,
    min_samples_leaf: usize = 1,
    criterion: Criterion = .gini,
};

/// A node in the decision tree.
pub const Node = struct {
    feature_index: ?usize,
    threshold: f64,
    left: ?*Node,
    right: ?*Node,
    prediction: f64,
    is_leaf: bool,
};

/// Classification decision tree.
pub const DecisionTree = struct {
    root: ?*Node,
    config: TreeConfig,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Creates an untrained decision tree.
    pub fn init(config: TreeConfig, allocator: std.mem.Allocator) Error!DecisionTree {
        return .{
            .root = null,
            .config = config,
            .allocator = allocator,
        };
    }

    /// Frees the tree recursively.
    pub fn deinit(self: *Self) void {
        destroy_node(self.root, self.allocator);
        self.root = null;
    }

    /// Builds the tree from all samples in `data`.
    pub fn fit(self: *Self, data: *Data(f64)) Error!void {
        if (data.nb_samples == 0) return;

        if (self.root != null) {
            destroy_node(self.root, self.allocator);
            self.root = null;
        }

        const x_rows = try self.allocator.alloc(usize, data.nb_samples);
        errdefer self.allocator.free(x_rows);
        for (0..data.nb_samples) |i| {
            x_rows[i] = i;
        }

        self.root = try build_tree(self, data, x_rows, data.y, 0);
        self.allocator.free(x_rows);
    }

    /// Predicts the label for a single sample.
    pub fn predict(self: *Self, x: []const f64) Error!f64 {
        const node = self.root orelse return error.NotFitted;
        return predict_node(node, x);
    }
};

fn destroy_node(node: ?*Node, allocator: std.mem.Allocator) void {
    const n = node orelse return;
    destroy_node(n.left, allocator);
    destroy_node(n.right, allocator);
    allocator.destroy(n);
}

fn predict_node(node: *Node, x: []const f64) Error!f64 {
    if (node.is_leaf) return node.prediction;
    const fi = node.feature_index orelse return error.ShapeMismatch;
    if (fi >= x.len) return error.ShapeMismatch;
    if (x[fi] <= node.threshold) {
        const left = node.left orelse return error.ShapeMismatch;
        return predict_node(left, x);
    } else {
        const right = node.right orelse return error.ShapeMismatch;
        return predict_node(right, x);
    }
}

/// Computes Gini impurity for a set of class labels.
fn gini(labels: []const f64) f64 {
    if (labels.len == 0) return 0.0;
    var impurity: f64 = 1.0;
    for (labels, 0..) |label, i| {
        if (already_seen(labels[0..i], label)) continue;
        const count = count_label(labels, label);
        const p = @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(labels.len));
        impurity -= p * p;
    }
    return impurity;
}

/// Computes entropy for a set of class labels.
fn entropy(labels: []const f64) f64 {
    if (labels.len == 0) return 0.0;
    var result: f64 = 0.0;
    for (labels, 0..) |label, i| {
        if (already_seen(labels[0..i], label)) continue;
        const count = count_label(labels, label);
        const p = @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(labels.len));
        result -= p * std.math.log2(p);
    }
    return result;
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

const Split = struct {
    feature_index: usize,
    threshold: f64,
    impurity: f64,
};

fn best_split(
    self: *DecisionTree,
    data: *Data(f64),
    x_rows: []const usize,
    y: []const f64,
) Error!?Split {
    if (x_rows.len < self.config.min_samples_split) return null;

    const parent_impurity = switch (self.config.criterion) {
        .gini => gini(y),
        .entropy => entropy(y),
    };

    var best: ?Split = null;
    var best_impurity: f64 = std.math.inf(f64);

    var values: std.ArrayList(f64) = .empty;
    defer values.deinit(self.allocator);

    var left_y_buf = try self.allocator.alloc(f64, x_rows.len);
    defer self.allocator.free(left_y_buf);
    var right_y_buf = try self.allocator.alloc(f64, x_rows.len);
    defer self.allocator.free(right_y_buf);

    for (0..data.nb_features) |feature_idx| {
        values.clearRetainingCapacity();
        for (x_rows) |row| {
            const val = try data.x.get(row, feature_idx);
            try values.append(self.allocator, val);
        }
        if (values.items.len < 2) continue;
        std.mem.sort(f64, values.items, {}, std.sort.asc(f64));

        for (0..values.items.len - 1) |i| {
            const threshold = (values.items[i] + values.items[i + 1]) / 2.0;

            var left_count: usize = 0;
            var right_count: usize = 0;
            for (x_rows, y) |row, label| {
                const val = try data.x.get(row, feature_idx);
                if (val <= threshold) {
                    left_y_buf[left_count] = label;
                    left_count += 1;
                } else {
                    right_y_buf[right_count] = label;
                    right_count += 1;
                }
            }

            if (left_count < self.config.min_samples_leaf or
                right_count < self.config.min_samples_leaf)
            {
                continue;
            }

            const left_impurity = switch (self.config.criterion) {
                .gini => gini(left_y_buf[0..left_count]),
                .entropy => entropy(left_y_buf[0..left_count]),
            };
            const right_impurity = switch (self.config.criterion) {
                .gini => gini(right_y_buf[0..right_count]),
                .entropy => entropy(right_y_buf[0..right_count]),
            };
            const weighted = (@as(f64, @floatFromInt(left_count)) * left_impurity +
                @as(f64, @floatFromInt(right_count)) * right_impurity) /
                @as(f64, @floatFromInt(x_rows.len));

            const info_gain = parent_impurity - weighted;
            if (info_gain > 0 and weighted < best_impurity) {
                best_impurity = weighted;
                best = .{
                    .feature_index = feature_idx,
                    .threshold = threshold,
                    .impurity = weighted,
                };
            }
        }
    }

    return best;
}

fn build_tree(
    self: *DecisionTree,
    data: *Data(f64),
    x_rows: []usize,
    y: []f64,
    depth: usize,
) Error!?*Node {
    const node = try self.allocator.create(Node);
    errdefer self.allocator.destroy(node);

    node.* = .{
        .feature_index = null,
        .threshold = 0.0,
        .left = null,
        .right = null,
        .prediction = 0.0,
        .is_leaf = false,
    };

    const parent_impurity = switch (self.config.criterion) {
        .gini => gini(y),
        .entropy => entropy(y),
    };

    const should_stop = depth >= self.config.max_depth or
        x_rows.len < self.config.min_samples_split or
        x_rows.len == 0 or
        parent_impurity < 1e-10;

    if (should_stop) {
        node.is_leaf = true;
        node.prediction = majority_class(y);
        return node;
    }

    const split = try best_split(self, data, x_rows, y);
    if (split == null) {
        node.is_leaf = true;
        node.prediction = majority_class(y);
        return node;
    }

    const s = split.?;

    var left_count: usize = 0;
    var right_count: usize = 0;
    for (x_rows) |row| {
        const val = try data.x.get(row, s.feature_index);
        if (val <= s.threshold) {
            left_count += 1;
        } else {
            right_count += 1;
        }
    }

    const left_x_rows = try self.allocator.alloc(usize, left_count);
    errdefer self.allocator.free(left_x_rows);
    const right_x_rows = try self.allocator.alloc(usize, right_count);
    errdefer self.allocator.free(right_x_rows);
    const left_y = try self.allocator.alloc(f64, left_count);
    errdefer self.allocator.free(left_y);
    const right_y = try self.allocator.alloc(f64, right_count);
    errdefer self.allocator.free(right_y);

    var li: usize = 0;
    var ri: usize = 0;
    for (x_rows, y) |row, label| {
        const val = try data.x.get(row, s.feature_index);
        if (val <= s.threshold) {
            left_x_rows[li] = row;
            left_y[li] = label;
            li += 1;
        } else {
            right_x_rows[ri] = row;
            right_y[ri] = label;
            ri += 1;
        }
    }

    node.feature_index = s.feature_index;
    node.threshold = s.threshold;
    node.is_leaf = false;

    node.left = try build_tree(self, data, left_x_rows, left_y, depth + 1);
    errdefer destroy_node(node.left, self.allocator);
    node.right = try build_tree(self, data, right_x_rows, right_y, depth + 1);
    errdefer destroy_node(node.right, self.allocator);

    // The children do not own these arrays; free them now that recursion succeeded.
    self.allocator.free(left_x_rows);
    self.allocator.free(left_y);
    self.allocator.free(right_x_rows);
    self.allocator.free(right_y);

    return node;
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

    {
        var tree = try DecisionTree.init(.{}, allocator);
        defer tree.deinit();
        try tree.fit(&data);
        try std.testing.expectApproxEqAbs(0.0, try tree.predict(&[_]f64{ 1.0, 1.0 }), 1e-12);
        try std.testing.expectApproxEqAbs(1.0, try tree.predict(&[_]f64{ 2.0, 2.0 }), 1e-12);
    }

    {
        var tree = try DecisionTree.init(.{ .criterion = .entropy }, allocator);
        defer tree.deinit();
        try tree.fit(&data);
        try std.testing.expectApproxEqAbs(0.0, try tree.predict(&[_]f64{ 1.0, 1.0 }), 1e-12);
        try std.testing.expectApproxEqAbs(1.0, try tree.predict(&[_]f64{ 2.0, 2.0 }), 1e-12);
    }
}
