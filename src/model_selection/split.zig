const std = @import("std");
const la = @import("../la.zig");
const Error = @import("../errors.zig").Error;

pub const TrainTestResult = struct {
    x_train: la.Matrix(f64),
    x_test: la.Matrix(f64),
    y_train: la.Vector(f64),
    y_test: la.Vector(f64),
};

pub const TrainTestSplitConfig = struct {
    test_size: f64 = 0.25,
    shuffle: bool = true,
    stratify: bool = false,
    random_seed: u64 = 42,
};

fn copyRows(dst: la.Matrix(f64), src: la.Matrix(f64), indices: []const usize) Error!void {
    for (indices, 0..) |orig_i, i| {
        for (0..src.cols) |j| {
            try dst.set(i, j, try src.get(orig_i, j));
        }
    }
}

fn copyValues(dst: la.Vector(f64), src: la.Vector(f64), indices: []const usize) Error!void {
    for (indices, 0..) |orig_i, i| {
        try dst.set(i, try src.get(orig_i));
    }
}

fn shuffleIndices(indices: []usize, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    rng.shuffle(usize, indices);
}

fn stratifiedIndices(y: la.Vector(f64), test_ratio: f64, seed: u64, allocator: std.mem.Allocator) Error![]usize {
    const n = y.len;

    var class_map = std.AutoHashMap(i64, std.ArrayList(usize)).init(allocator);
    defer {
        var it = class_map.valueIterator();
        while (it.next()) |list| list.deinit(allocator);
        class_map.deinit();
    }

    for (0..n) |i| {
        const val = try y.get(i);
        const key = @as(i64, @intFromFloat(val));
        const gop = try class_map.getOrPut(key);
        if (!gop.found_existing) {
            gop.value_ptr.* = std.ArrayList(usize).empty;
        }
        try gop.value_ptr.append(allocator, i);
    }

    var train: std.ArrayList(usize) = .empty;
    defer train.deinit(allocator);
    var test_indices: std.ArrayList(usize) = .empty;
    defer test_indices.deinit(allocator);

    var it = class_map.valueIterator();
    while (it.next()) |list| {
        const class_n = list.items.len;
        if (class_n == 1) {
            try train.append(allocator, list.items[0]);
            continue;
        }

        var class_idx = try allocator.alloc(usize, class_n);
        defer allocator.free(class_idx);
        @memcpy(class_idx, list.items);
        shuffleIndices(class_idx, seed);

        const n_test = @max(1, @min(class_n - 1, @as(usize, @intFromFloat(@round(@as(f64, @floatFromInt(class_n)) * test_ratio)))));
        const n_train = class_n - n_test;

        try train.appendSlice(allocator, class_idx[0..n_train]);
        try test_indices.appendSlice(allocator, class_idx[n_train..]);
    }

    shuffleIndices(train.items, seed);
    shuffleIndices(test_indices.items, seed +% 1);

    var result = try allocator.alloc(usize, n);
    @memcpy(result[0..train.items.len], train.items);
    @memcpy(result[train.items.len..], test_indices.items);
    return result;
}

pub fn train_test_split(x: la.Matrix(f64), y: la.Vector(f64), config: TrainTestSplitConfig, allocator: std.mem.Allocator) Error!TrainTestResult {
    if (x.rows != y.len) return error.ShapeMismatch;
    if (x.rows == 0) return error.InvalidDimension;
    if (config.test_size <= 0.0 or config.test_size >= 1.0) return error.InvalidDimension;

    const n = x.rows;
    const n_test = @max(1, @min(n - 1, @as(usize, @intFromFloat(@round(@as(f64, @floatFromInt(n)) * config.test_size)))));
    const n_train = n - n_test;

    if (n_train == 0 or n_test == 0) return error.InvalidDimension;

    var indices = try allocator.alloc(usize, n);
    defer allocator.free(indices);
    for (0..n) |i| indices[i] = i;

    if (config.stratify) {
        const strat = try stratifiedIndices(y, config.test_size, config.random_seed, allocator);
        defer allocator.free(strat);
        @memcpy(indices, strat);
    } else if (config.shuffle) {
        shuffleIndices(indices, config.random_seed);
    }

    var x_train = try la.Matrix(f64).init(allocator, n_train, x.cols);
    errdefer x_train.deinit(allocator);
    var x_test = try la.Matrix(f64).init(allocator, n_test, x.cols);
    errdefer x_test.deinit(allocator);
    var y_train = try la.Vector(f64).init(allocator, n_train);
    errdefer y_train.deinit(allocator);
    var y_test = try la.Vector(f64).init(allocator, n_test);
    errdefer y_test.deinit(allocator);

    try copyRows(x_train, x, indices[0..n_train]);
    try copyRows(x_test, x, indices[n_train..]);
    try copyValues(y_train, y, indices[0..n_train]);
    try copyValues(y_test, y, indices[n_train..]);

    return .{
        .x_train = x_train,
        .x_test = x_test,
        .y_train = y_train,
        .y_test = y_test,
    };
}

pub const Fold = struct {
    x_train: la.Matrix(f64),
    x_test: la.Matrix(f64),
    y_train: la.Vector(f64),
    y_test: la.Vector(f64),

    pub fn deinit(self: *Fold, allocator: std.mem.Allocator) void {
        self.x_train.deinit(allocator);
        self.x_test.deinit(allocator);
        self.y_train.deinit(allocator);
        self.y_test.deinit(allocator);
    }
};

pub fn k_fold_split(x: la.Matrix(f64), y: la.Vector(f64), n_folds: usize, shuffle: bool, seed: u64, allocator: std.mem.Allocator) Error![]Fold {
    if (x.rows != y.len) return error.ShapeMismatch;
    if (x.rows == 0) return error.InvalidDimension;
    if (n_folds < 2) return error.InvalidDimension;
    if (x.rows < n_folds) return error.InvalidDimension;

    const n = x.rows;
    var indices = try allocator.alloc(usize, n);
    defer allocator.free(indices);
    for (0..n) |i| indices[i] = i;

    if (shuffle) shuffleIndices(indices, seed);

    const fold_size = n / n_folds;
    const remainder = n % n_folds;

    var folds = try allocator.alloc(Fold, n_folds);
    errdefer {
        for (folds) |*fold| fold.deinit(allocator);
        allocator.free(folds);
    }

    var start: usize = 0;
    for (0..n_folds) |i| {
        const end = start + fold_size + @intFromBool(i < remainder);
        const n_test = end - start;
        const n_train = n - n_test;

        var x_train = try la.Matrix(f64).init(allocator, n_train, x.cols);
        errdefer x_train.deinit(allocator);
        var x_test = try la.Matrix(f64).init(allocator, n_test, x.cols);
        errdefer x_test.deinit(allocator);
        var y_train = try la.Vector(f64).init(allocator, n_train);
        errdefer y_train.deinit(allocator);
        var y_test = try la.Vector(f64).init(allocator, n_test);
        errdefer y_test.deinit(allocator);

        var train_idx = try allocator.alloc(usize, n_train);
        defer allocator.free(train_idx);
        @memcpy(train_idx[0..start], indices[0..start]);
        @memcpy(train_idx[start..], indices[end..n]);

        try copyRows(x_train, x, train_idx);
        try copyRows(x_test, x, indices[start..end]);
        try copyValues(y_train, y, train_idx);
        try copyValues(y_test, y, indices[start..end]);

        folds[i] = .{
            .x_train = x_train,
            .x_test = x_test,
            .y_train = y_train,
            .y_test = y_test,
        };

        start = end;
    }

    return folds;
}

pub fn deinitFoldSlice(folds: []Fold, allocator: std.mem.Allocator) void {
    for (folds) |*fold| fold.deinit(allocator);
    allocator.free(folds);
}

test "train_test_split basic" {
    const allocator = std.testing.allocator;
    const T = f64;
    const V = la.Vector(T);
    const M = la.Matrix(T);

    var x = try M.fromRowSlice(allocator, 8, 2, &[_]T{
        1.0,  2.0,
        3.0,  4.0,
        5.0,  6.0,
        7.0,  8.0,
        9.0,  10.0,
        11.0, 12.0,
        13.0, 14.0,
        15.0, 16.0,
    });
    defer x.deinit(allocator);

    var y = try V.fromSlice(allocator, &[_]T{ 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0 });
    defer y.deinit(allocator);

    var result = try train_test_split(x, y, .{}, allocator);
    defer {
        result.x_train.deinit(allocator);
        result.x_test.deinit(allocator);
        result.y_train.deinit(allocator);
        result.y_test.deinit(allocator);
    }

    try std.testing.expectEqual(6, result.x_train.rows);
    try std.testing.expectEqual(2, result.x_test.rows);
}

test "train_test_split no shuffle" {
    const allocator = std.testing.allocator;
    const T = f64;
    const V = la.Vector(T);
    const M = la.Matrix(T);

    var x = try M.fromRowSlice(allocator, 4, 2, &[_]T{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
    });
    defer x.deinit(allocator);

    var y = try V.fromSlice(allocator, &[_]T{ 0.0, 1.0, 2.0, 3.0 });
    defer y.deinit(allocator);

    var result = try train_test_split(x, y, .{ .shuffle = false }, allocator);
    defer {
        result.x_train.deinit(allocator);
        result.x_test.deinit(allocator);
        result.y_train.deinit(allocator);
        result.y_test.deinit(allocator);
    }

    try std.testing.expectEqual(3, result.x_train.rows);
    try std.testing.expectEqual(1, result.x_test.rows);
    try std.testing.expectEqual(1.0, try result.x_train.get(0, 0));
    try std.testing.expectEqual(7.0, try result.x_test.get(0, 0));
}

test "train_test_split stratify" {
    const allocator = std.testing.allocator;
    const T = f64;
    const V = la.Vector(T);
    const M = la.Matrix(T);

    var x = try M.fromRowSlice(allocator, 8, 1, &[_]T{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 });
    defer x.deinit(allocator);

    var y = try V.fromSlice(allocator, &[_]T{ 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0 });
    defer y.deinit(allocator);

    var result = try train_test_split(x, y, .{ .stratify = true, .shuffle = false }, allocator);
    defer {
        result.x_train.deinit(allocator);
        result.x_test.deinit(allocator);
        result.y_train.deinit(allocator);
        result.y_test.deinit(allocator);
    }

    try std.testing.expect(result.x_train.rows + result.x_test.rows == 8);
}

test "k_fold_split basic" {
    const allocator = std.testing.allocator;
    const T = f64;
    const V = la.Vector(T);
    const M = la.Matrix(T);

    var x = try M.fromRowSlice(allocator, 6, 2, &[_]T{
        1.0,  2.0,
        3.0,  4.0,
        5.0,  6.0,
        7.0,  8.0,
        9.0,  10.0,
        11.0, 12.0,
    });
    defer x.deinit(allocator);

    var y = try V.fromSlice(allocator, &[_]T{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 });
    defer y.deinit(allocator);

    const folds = try k_fold_split(x, y, 3, false, 0, allocator);
    defer deinitFoldSlice(folds, allocator);

    try std.testing.expectEqual(3, folds.len);
    for (folds) |fold| {
        try std.testing.expectEqual(4, fold.x_train.rows);
        try std.testing.expectEqual(2, fold.x_test.rows);
    }
}
