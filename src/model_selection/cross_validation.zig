const std = @import("std");
const Error = @import("../errors.zig").Error;

/// Shuffle a slice of indices in-place using a deterministic PRNG.
fn shuffleIndices(indices: []usize, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    rng.shuffle(usize, indices);
}

/// Result type returned by cross-validation iterators. The caller owns both
/// `train_indices` and `test_indices` slices and must free them with the
/// allocator that was passed to the iterator's `init`.
pub const SplitIndices = struct {
    train_indices: []usize,
    test_indices: []usize,
};

/// K-Fold cross-validation iterator.
///
/// Splits `n_samples` indices into `n_folds` consecutive folds. Each call to
/// `next()` returns one fold where the test fold is held out and the remaining
/// samples form the training set.
pub const KFold = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    indices: []usize,
    n_folds: usize,
    current: usize,

    /// Create a new KFold iterator.
    /// `allocator` is used for internal state and for allocations returned
    /// by `next()`. Caller must call `deinit()` when done.
    pub fn init(allocator: std.mem.Allocator, n_samples: usize, n_folds: usize, shuffle: bool, seed: u64) Error!Self {
        if (n_folds < 2) return error.InvalidDimension;
        if (n_samples < n_folds) return error.InvalidDimension;

        const indices = try allocator.alloc(usize, n_samples);
        errdefer allocator.free(indices);
        for (0..n_samples) |i| indices[i] = i;

        if (shuffle) shuffleIndices(indices, seed);

        return .{
            .allocator = allocator,
            .indices = indices,
            .n_folds = n_folds,
            .current = 0,
        };
    }

    /// Free internal state. This does **not** free slices previously
    /// returned by `next()`.
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.indices);
        self.* = undefined;
    }

    /// Return the next train/test split, or `null` when all folds have
    /// been consumed. Returns `error.OutOfMemory` on allocation failure.
    pub fn next(self: *Self) Error!?SplitIndices {
        if (self.current >= self.n_folds) return null;

        const n = self.indices.len;
        const fold_size = n / self.n_folds;
        const remainder = n % self.n_folds;

        var start: usize = 0;
        for (0..self.current) |i| {
            start += fold_size + @intFromBool(i < remainder);
        }
        const end = start + fold_size + @intFromBool(self.current < remainder);
        const n_test = end - start;
        const n_train = n - n_test;

        const test_indices = try self.allocator.alloc(usize, n_test);
        errdefer self.allocator.free(test_indices);
        @memcpy(test_indices, self.indices[start..end]);

        const train_indices = try self.allocator.alloc(usize, n_train);
        @memcpy(train_indices[0..start], self.indices[0..start]);
        @memcpy(train_indices[start..], self.indices[end..n]);

        self.current += 1;
        return .{ .train_indices = train_indices, .test_indices = test_indices };
    }
};

/// Stratified K-Fold cross-validation iterator.
///
/// Preserves the proportion of each class label across folds. Labels are
/// grouped by equality, shuffled within each group, and distributed across
/// folds before the train/test indices are assembled.
pub fn StratifiedKFold(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        n_samples: usize,
        n_folds: usize,
        /// Concatenation of per-fold test indices, grouped by fold.
        test_by_fold: []usize,
        /// Offset in `test_by_fold` where each fold's test indices begin.
        fold_offsets: []usize,
        current: usize,

        pub fn init(allocator: std.mem.Allocator, labels: []const T, n_folds: usize, shuffle: bool, seed: u64) Error!Self {
            const n_samples = labels.len;
            if (n_folds < 2) return error.InvalidDimension;
            if (n_samples < n_folds) return error.InvalidDimension;

            // Per-fold test lists.
            var fold_tests = try allocator.alloc(std.ArrayList(usize), n_folds);
            errdefer allocator.free(fold_tests);
            for (0..n_folds) |i| fold_tests[i] = std.ArrayList(usize).empty;
            errdefer {
                for (0..n_folds) |i| fold_tests[i].deinit(allocator);
            }

            // Group sample indices by label using a sorted (label, index) array.
            // This avoids requiring T to be hashable.
            const Pair = struct { label: T, index: usize };
            var pairs = try allocator.alloc(Pair, n_samples);
            defer allocator.free(pairs);
            for (0..n_samples) |i| pairs[i] = .{ .label = labels[i], .index = i };

            const SortContext = struct {
                pub fn lessThan(_: @This(), a: Pair, b: Pair) bool {
                    return a.label < b.label;
                }
            };
            std.mem.sort(Pair, pairs, SortContext{}, SortContext.lessThan);

            var group_start: usize = 0;
            while (group_start < n_samples) {
                var group_end = group_start + 1;
                while (group_end < n_samples and pairs[group_end].label == pairs[group_start].label) {
                    group_end += 1;
                }
                const class_n = group_end - group_start;
                var class_idx = try allocator.alloc(usize, class_n);
                defer allocator.free(class_idx);
                for (0..class_n) |k| class_idx[k] = pairs[group_start + k].index;
                if (shuffle) shuffleIndices(class_idx, seed);

                const fold_size = class_n / n_folds;
                const remainder = class_n % n_folds;
                var start: usize = 0;
                for (0..n_folds) |i| {
                    const end = start + fold_size + @intFromBool(i < remainder);
                    try fold_tests[i].appendSlice(allocator, class_idx[start..end]);
                    start = end;
                }

                group_start = group_end;
            }

            // Flatten per-fold test indices and record offsets.
            var total_test: usize = 0;
            for (fold_tests) |list| total_test += list.items.len;

            const test_by_fold = try allocator.alloc(usize, total_test);
            errdefer allocator.free(test_by_fold);
            const fold_offsets = try allocator.alloc(usize, n_folds + 1);
            errdefer allocator.free(fold_offsets);

            fold_offsets[0] = 0;
            var offset: usize = 0;
            for (0..n_folds) |i| {
                @memcpy(test_by_fold[offset..][0..fold_tests[i].items.len], fold_tests[i].items);
                offset += fold_tests[i].items.len;
                fold_offsets[i + 1] = offset;
                fold_tests[i].deinit(allocator);
            }
            allocator.free(fold_tests);

            return .{
                .allocator = allocator,
                .n_samples = n_samples,
                .n_folds = n_folds,
                .test_by_fold = test_by_fold,
                .fold_offsets = fold_offsets,
                .current = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.test_by_fold);
            self.allocator.free(self.fold_offsets);
            self.* = undefined;
        }

        pub fn next(self: *Self) Error!?SplitIndices {
            if (self.current >= self.n_folds) return null;

            const test_start = self.fold_offsets[self.current];
            const test_end = self.fold_offsets[self.current + 1];
            const n_test = test_end - test_start;
            const n_train = self.n_samples - n_test;

            const test_indices = try self.allocator.alloc(usize, n_test);
            errdefer self.allocator.free(test_indices);
            @memcpy(test_indices, self.test_by_fold[test_start..test_end]);

            const train_indices = try self.allocator.alloc(usize, n_train);

            // Mark test samples so we can build the training set.
            const is_test = try self.allocator.alloc(bool, self.n_samples);
            defer self.allocator.free(is_test);
            @memset(is_test, false);
            for (self.test_by_fold[test_start..test_end]) |idx| is_test[idx] = true;

            var pos: usize = 0;
            for (0..self.n_samples) |i| {
                if (!is_test[i]) {
                    train_indices[pos] = i;
                    pos += 1;
                }
            }
            std.debug.assert(pos == n_train);

            self.current += 1;
            return .{ .train_indices = train_indices, .test_indices = test_indices };
        }
    };
}

/// Leave-one-out cross-validation iterator.
///
/// Produces `n_samples` splits where each test set contains exactly one sample
/// and the training set contains all remaining samples.
pub const LeaveOneOut = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    n_samples: usize,
    current: usize,

    pub fn init(allocator: std.mem.Allocator, n_samples: usize) Error!Self {
        if (n_samples == 0) return error.InvalidDimension;
        return .{
            .allocator = allocator,
            .n_samples = n_samples,
            .current = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.* = undefined;
    }

    pub fn next(self: *Self) Error!?SplitIndices {
        if (self.current >= self.n_samples) return null;

        const n_train = self.n_samples - 1;
        const test_indices = try self.allocator.alloc(usize, 1);
        errdefer self.allocator.free(test_indices);
        test_indices[0] = self.current;

        const train_indices = try self.allocator.alloc(usize, n_train);
        var pos: usize = 0;
        for (0..self.n_samples) |i| {
            if (i != self.current) {
                train_indices[pos] = i;
                pos += 1;
            }
        }

        self.current += 1;
        return .{ .train_indices = train_indices, .test_indices = test_indices };
    }
};

/// Shuffle-split cross-validation iterator.
///
/// Produces `n_splits` random train/test splits. In each split a fraction
/// `test_size` of the samples is assigned to the test set.
pub const ShuffleSplit = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    n_samples: usize,
    n_splits: usize,
    n_test: usize,
    current: usize,
    seed: u64,

    pub fn init(allocator: std.mem.Allocator, n_samples: usize, n_splits: usize, test_size: f64, seed: u64) Error!Self {
        if (n_splits == 0) return error.InvalidDimension;
        if (n_samples == 0) return error.InvalidDimension;
        if (test_size <= 0.0 or test_size >= 1.0) return error.InvalidDimension;

        const n_test = @max(1, @min(n_samples - 1, @as(usize, @intFromFloat(@round(@as(f64, @floatFromInt(n_samples)) * test_size)))));
        const n_train = n_samples - n_test;
        if (n_train == 0 or n_test == 0) return error.InvalidDimension;

        return .{
            .allocator = allocator,
            .n_samples = n_samples,
            .n_splits = n_splits,
            .n_test = n_test,
            .current = 0,
            .seed = seed,
        };
    }

    pub fn deinit(self: *Self) void {
        self.* = undefined;
    }

    pub fn next(self: *Self) Error!?SplitIndices {
        if (self.current >= self.n_splits) return null;

        const n = self.n_samples;
        const n_train = n - self.n_test;

        var indices = try self.allocator.alloc(usize, n);
        defer self.allocator.free(indices);
        for (0..n) |i| indices[i] = i;
        shuffleIndices(indices, self.seed +% self.current);

        const test_indices = try self.allocator.alloc(usize, self.n_test);
        errdefer self.allocator.free(test_indices);
        @memcpy(test_indices, indices[n_train..n]);

        const train_indices = try self.allocator.alloc(usize, n_train);
        @memcpy(train_indices, indices[0..n_train]);

        self.current += 1;
        return .{ .train_indices = train_indices, .test_indices = test_indices };
    }
};

fn freeSplit(allocator: std.mem.Allocator, split_indices: SplitIndices) void {
    allocator.free(split_indices.train_indices);
    allocator.free(split_indices.test_indices);
}

test "KFold basic" {
    const allocator = std.testing.allocator;
    var kf = try KFold.init(allocator, 100, 5, false, 42);
    defer kf.deinit();

    var count: usize = 0;
    while (try kf.next()) |split_indices| {
        defer freeSplit(allocator, split_indices);
        try std.testing.expectEqual(20, split_indices.test_indices.len);
        try std.testing.expectEqual(80, split_indices.train_indices.len);
        count += 1;
    }
    try std.testing.expectEqual(5, count);
}

test "KFold uneven sample count" {
    const allocator = std.testing.allocator;
    var kf = try KFold.init(allocator, 10, 3, false, 42);
    defer kf.deinit();

    var total_test: usize = 0;
    while (try kf.next()) |split_indices| {
        defer freeSplit(allocator, split_indices);
        total_test += split_indices.test_indices.len;
    }
    try std.testing.expectEqual(10, total_test);
}

test "KFold covers all indices exactly once" {
    const allocator = std.testing.allocator;
    var kf = try KFold.init(allocator, 50, 5, false, 42);
    defer kf.deinit();

    var seen = std.AutoHashMap(usize, void).init(allocator);
    defer seen.deinit();

    while (try kf.next()) |split_indices| {
        defer freeSplit(allocator, split_indices);
        for (split_indices.test_indices) |idx| {
            try std.testing.expect(!seen.contains(idx));
            try seen.put(idx, {});
        }
    }
    try std.testing.expectEqual(50, seen.count());
}

test "KFold train and test are disjoint" {
    const allocator = std.testing.allocator;
    var kf = try KFold.init(allocator, 30, 3, false, 42);
    defer kf.deinit();

    while (try kf.next()) |split_indices| {
        defer freeSplit(allocator, split_indices);
        for (split_indices.train_indices) |t| {
            for (split_indices.test_indices) |s| {
                try std.testing.expect(t != s);
            }
        }
    }
}

test "StratifiedKFold preserves class proportions" {
    const allocator = std.testing.allocator;
    const labels = [_]f64{ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var skf = try StratifiedKFold(f64).init(allocator, &labels, 5, false, 42);
    defer skf.deinit();

    var count: usize = 0;
    while (try skf.next()) |split_indices| {
        defer freeSplit(allocator, split_indices);
        try std.testing.expectEqual(2, split_indices.test_indices.len);
        try std.testing.expectEqual(8, split_indices.train_indices.len);

        var class_0: usize = 0;
        var class_1: usize = 0;
        for (split_indices.test_indices) |idx| {
            if (labels[idx] == 0.0) class_0 += 1 else class_1 += 1;
        }
        try std.testing.expectEqual(1, class_0);
        try std.testing.expectEqual(1, class_1);
        count += 1;
    }
    try std.testing.expectEqual(5, count);
}

test "StratifiedKFold covers all indices exactly once" {
    const allocator = std.testing.allocator;
    const labels = [_]i32{ 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 };
    var skf = try StratifiedKFold(i32).init(allocator, &labels, 4, true, 123);
    defer skf.deinit();

    var seen = std.AutoHashMap(usize, void).init(allocator);
    defer seen.deinit();

    while (try skf.next()) |split_indices| {
        defer freeSplit(allocator, split_indices);
        for (split_indices.test_indices) |idx| {
            try std.testing.expect(!seen.contains(idx));
            try seen.put(idx, {});
        }
    }
    try std.testing.expectEqual(labels.len, seen.count());
}

test "LeaveOneOut" {
    const allocator = std.testing.allocator;
    var loo = try LeaveOneOut.init(allocator, 5);
    defer loo.deinit();

    var i: usize = 0;
    while (try loo.next()) |split_indices| {
        defer freeSplit(allocator, split_indices);
        try std.testing.expectEqual(1, split_indices.test_indices.len);
        try std.testing.expectEqual(i, split_indices.test_indices[0]);
        try std.testing.expectEqual(4, split_indices.train_indices.len);
        i += 1;
    }
    try std.testing.expectEqual(5, i);
}

test "ShuffleSplit basic" {
    const allocator = std.testing.allocator;
    var ss = try ShuffleSplit.init(allocator, 100, 3, 0.2, 42);
    defer ss.deinit();

    var count: usize = 0;
    while (try ss.next()) |split_indices| {
        defer freeSplit(allocator, split_indices);
        try std.testing.expectEqual(20, split_indices.test_indices.len);
        try std.testing.expectEqual(80, split_indices.train_indices.len);
        count += 1;
    }
    try std.testing.expectEqual(3, count);
}

test "ShuffleSplit different seeds give different splits" {
    const allocator = std.testing.allocator;
    var ss = try ShuffleSplit.init(allocator, 100, 3, 0.2, 42);
    defer ss.deinit();

    var first: ?[]usize = null;
    defer if (first) |f| allocator.free(f);

    while (try ss.next()) |split_indices| {
        defer allocator.free(split_indices.train_indices);
        if (first == null) {
            first = try allocator.dupe(usize, split_indices.test_indices);
        } else {
            try std.testing.expect(!std.mem.eql(usize, first.?, split_indices.test_indices));
        }
        allocator.free(split_indices.test_indices);
    }
}

test "KFold rejects invalid arguments" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.InvalidDimension, KFold.init(allocator, 10, 1, false, 42));
    try std.testing.expectError(error.InvalidDimension, KFold.init(allocator, 3, 5, false, 42));
}

test "ShuffleSplit rejects invalid arguments" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.InvalidDimension, ShuffleSplit.init(allocator, 100, 1, 0.0, 42));
    try std.testing.expectError(error.InvalidDimension, ShuffleSplit.init(allocator, 100, 1, 1.0, 42));
}
