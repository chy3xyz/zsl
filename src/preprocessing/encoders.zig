const std = @import("std");
const Error = @import("../errors.zig").Error;
const la = @import("../la.zig");

fn findCategory(categories: []const []const u8, value: []const u8) ?usize {
    for (categories, 0..) |cat, i| {
        if (std.mem.eql(u8, cat, value)) return i;
    }
    return null;
}

fn fitColumn(allocator: std.mem.Allocator, column: []const []const u8) Error![]const []const u8 {
    var cats = std.ArrayList([]const u8).empty;
    errdefer {
        for (cats.items) |s| allocator.free(s);
        cats.deinit(allocator);
    }
    var seen = std.StringHashMap(void).init(allocator);
    defer seen.deinit();
    for (column) |value| {
        const result = try seen.getOrPut(value);
        if (!result.found_existing) {
            const copy = try allocator.dupe(u8, value);
            errdefer allocator.free(copy);
            try cats.append(allocator, copy);
            errdefer _ = cats.pop();
        }
    }
    return try cats.toOwnedSlice(allocator);
}

/// LabelEncoder encodes categorical labels with integer values between 0 and
/// n_classes-1, preserving the order of first appearance.
pub const LabelEncoder = struct {
    allocator: std.mem.Allocator,
    fitted: bool,
    classes_: [][]const u8,
    class_to_idx: std.StringHashMap(usize),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .fitted = false,
            .classes_ = &[_][]const u8{},
            .class_to_idx = std.StringHashMap(usize).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.classes_) |class| self.allocator.free(class);
        self.allocator.free(self.classes_);
        self.classes_ = &[_][]const u8{};
        self.class_to_idx.deinit();
        self.class_to_idx = std.StringHashMap(usize).init(self.allocator);
        self.fitted = false;
    }

    pub fn fit(self: *Self, labels: []const []const u8) Error!void {
        if (labels.len == 0) return error.InvalidDimension;

        self.deinit();

        var unique_refs = std.ArrayList([]const u8).empty;
        defer unique_refs.deinit(self.allocator);
        var seen = std.StringHashMap(void).init(self.allocator);
        defer seen.deinit();
        for (labels) |label| {
            const result = try seen.getOrPut(label);
            if (!result.found_existing) {
                try unique_refs.append(self.allocator, label);
            }
        }

        self.classes_ = try self.allocator.alloc([]const u8, unique_refs.items.len);
        errdefer self.allocator.free(self.classes_);
        @memset(self.classes_, &[_]u8{});
        errdefer {
            for (self.classes_) |class| self.allocator.free(class);
        }
        for (unique_refs.items, 0..) |label, i| {
            self.classes_[i] = try self.allocator.dupe(u8, label);
        }

        for (self.classes_, 0..) |class, i| {
            try self.class_to_idx.put(class, i);
        }
        self.fitted = true;
    }

    pub fn transform(self: *Self, labels: []const []const u8) Error![]usize {
        if (!self.fitted) return error.NotFitted;

        const result = try self.allocator.alloc(usize, labels.len);
        errdefer self.allocator.free(result);
        for (labels, 0..) |label, i| {
            result[i] = self.class_to_idx.get(label) orelse return error.InvalidDimension;
        }
        return result;
    }

    pub fn fit_transform(self: *Self, labels: []const []const u8) Error![]usize {
        try self.fit(labels);
        return try self.transform(labels);
    }

    pub fn inverse_transform(self: *Self, codes: []const usize, allocator: std.mem.Allocator) Error![]const []const u8 {
        if (!self.fitted) return error.NotFitted;

        const result = try allocator.alloc([]const u8, codes.len);
        errdefer allocator.free(result);
        for (codes, 0..) |code, i| {
            if (code >= self.classes_.len) return error.IndexOutOfBounds;
            result[i] = try allocator.dupe(u8, self.classes_[code]);
        }
        return result;
    }
};

/// OrdinalEncoder encodes each categorical feature column as ordinal integers.
/// Input columns are passed in feature-major order: each element of `columns`
/// is one feature's values across all samples.
pub const OrdinalEncoder = struct {
    allocator: std.mem.Allocator,
    fitted: bool,
    categories_: [][]const []const u8,
    n_features: usize,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .fitted = false,
            .categories_ = &[_][]const []const u8{},
            .n_features = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.categories_) |cat| {
            for (cat) |s| self.allocator.free(s);
            self.allocator.free(cat);
        }
        self.allocator.free(self.categories_);
        self.categories_ = &[_][]const []const u8{};
        self.n_features = 0;
        self.fitted = false;
    }

    pub fn fit(self: *Self, columns: []const []const []const u8) Error!void {
        if (columns.len == 0) return error.InvalidDimension;

        self.deinit();
        self.n_features = columns.len;
        self.categories_ = try self.allocator.alloc([]const []const u8, self.n_features);
        errdefer self.allocator.free(self.categories_);
        @memset(self.categories_, &[_][]const u8{});
        var populated: usize = 0;
        errdefer {
            for (0..populated) |j| {
                for (self.categories_[j]) |s| self.allocator.free(s);
                self.allocator.free(self.categories_[j]);
            }
        }
        for (columns, 0..) |column, j| {
            if (column.len == 0) return error.InvalidDimension;
            self.categories_[j] = try fitColumn(self.allocator, column);
            populated += 1;
        }
        self.fitted = true;
    }

    pub fn transform(self: *Self, columns: []const []const []const u8) Error!la.Matrix(usize) {
        if (!self.fitted) return error.NotFitted;
        if (columns.len != self.n_features) return error.ShapeMismatch;

        const n_samples = columns[0].len;
        for (columns) |col| {
            if (col.len != n_samples) return error.ShapeMismatch;
        }

        var result = try la.Matrix(usize).init(self.allocator, n_samples, self.n_features);
        errdefer result.deinit(self.allocator);
        for (0..self.n_features) |j| {
            for (0..n_samples) |i| {
                const idx = findCategory(self.categories_[j], columns[j][i]) orelse return error.InvalidDimension;
                try result.set(i, j, idx);
            }
        }
        return result;
    }

    pub fn fit_transform(self: *Self, columns: []const []const []const u8) Error!la.Matrix(usize) {
        try self.fit(columns);
        return try self.transform(columns);
    }

    pub fn inverse_transform(self: *Self, codes: la.Matrix(usize), allocator: std.mem.Allocator) Error![]const []const []const u8 {
        if (!self.fitted) return error.NotFitted;
        if (codes.cols != self.n_features) return error.ShapeMismatch;

        const n_samples = codes.rows;
        var columns = try allocator.alloc([]const []const u8, self.n_features);
        errdefer allocator.free(columns);
        @memset(columns, &[_][]const u8{});
        var populated: usize = 0;
        errdefer {
            for (0..populated) |j| {
                for (columns[j]) |s| allocator.free(s);
                allocator.free(columns[j]);
            }
        }
        for (0..self.n_features) |j| {
            var col = try allocator.alloc([]const u8, n_samples);
            errdefer {
                for (col) |s| allocator.free(s);
                allocator.free(col);
            }
            for (0..n_samples) |i| {
                const code = try codes.get(i, j);
                if (code >= self.categories_[j].len) return error.IndexOutOfBounds;
                col[i] = try allocator.dupe(u8, self.categories_[j][code]);
            }
            columns[j] = col;
            populated += 1;
        }
        return columns;
    }
};

/// OneHotEncoder encodes categorical features as a one-hot numeric matrix.
/// Input columns are passed in feature-major order: each element of `columns`
/// is one feature's values across all samples.
pub const OneHotEncoder = struct {
    allocator: std.mem.Allocator,
    fitted: bool,
    categories_: [][]const []const u8,
    n_features: usize,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .fitted = false,
            .categories_ = &[_][]const []const u8{},
            .n_features = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.categories_) |cat| {
            for (cat) |s| self.allocator.free(s);
            self.allocator.free(cat);
        }
        self.allocator.free(self.categories_);
        self.categories_ = &[_][]const []const u8{};
        self.n_features = 0;
        self.fitted = false;
    }

    pub fn fit(self: *Self, columns: []const []const []const u8) Error!void {
        if (columns.len == 0) return error.InvalidDimension;

        self.deinit();
        self.n_features = columns.len;
        self.categories_ = try self.allocator.alloc([]const []const u8, self.n_features);
        errdefer self.allocator.free(self.categories_);
        @memset(self.categories_, &[_][]const u8{});
        var populated: usize = 0;
        errdefer {
            for (0..populated) |j| {
                for (self.categories_[j]) |s| self.allocator.free(s);
                self.allocator.free(self.categories_[j]);
            }
        }
        for (columns, 0..) |column, j| {
            if (column.len == 0) return error.InvalidDimension;
            self.categories_[j] = try fitColumn(self.allocator, column);
            populated += 1;
        }
        self.fitted = true;
    }

    pub fn transform(self: *Self, columns: []const []const []const u8) Error!la.Matrix(f64) {
        if (!self.fitted) return error.NotFitted;
        if (columns.len != self.n_features) return error.ShapeMismatch;

        const n_samples = columns[0].len;
        for (columns) |col| {
            if (col.len != n_samples) return error.ShapeMismatch;
        }

        var n_output_cols: usize = 0;
        for (self.categories_) |cats| n_output_cols += cats.len;

        var result = try la.Matrix(f64).init(self.allocator, n_samples, n_output_cols);
        errdefer result.deinit(self.allocator);
        @memset(result.data, 0);

        for (0..n_samples) |i| {
            var offset: usize = 0;
            for (0..self.n_features) |j| {
                const idx = findCategory(self.categories_[j], columns[j][i]) orelse return error.InvalidDimension;
                try result.set(i, offset + idx, 1.0);
                offset += self.categories_[j].len;
            }
        }
        return result;
    }

    pub fn fit_transform(self: *Self, columns: []const []const []const u8) Error!la.Matrix(f64) {
        try self.fit(columns);
        return try self.transform(columns);
    }

    pub fn inverse_transform(self: *Self, encoded: la.Matrix(f64), allocator: std.mem.Allocator) Error![]const []const []const u8 {
        if (!self.fitted) return error.NotFitted;

        const n_samples = encoded.rows;
        var columns = try allocator.alloc([]const []const u8, self.n_features);
        errdefer allocator.free(columns);
        @memset(columns, &[_][]const u8{});
        var populated: usize = 0;
        errdefer {
            for (0..populated) |j| {
                for (columns[j]) |s| allocator.free(s);
                allocator.free(columns[j]);
            }
        }

        for (0..self.n_features) |j| {
            var col = try allocator.alloc([]const u8, n_samples);
            errdefer {
                for (col) |s| allocator.free(s);
                allocator.free(col);
            }
            var offset: usize = 0;
            for (0..j) |k| offset += self.categories_[k].len;
            const n_cats = self.categories_[j].len;
            for (0..n_samples) |i| {
                var max_idx: usize = 0;
                var max_val: f64 = try encoded.get(i, offset);
                for (1..n_cats) |k| {
                    const val = try encoded.get(i, offset + k);
                    if (val > max_val) {
                        max_val = val;
                        max_idx = k;
                    }
                }
                col[i] = try allocator.dupe(u8, self.categories_[j][max_idx]);
            }
            columns[j] = col;
            populated += 1;
        }
        return columns;
    }
};

test "LabelEncoder fit_transform and inverse_transform" {
    var encoder = LabelEncoder.init(std.testing.allocator);
    defer encoder.deinit();

    const labels = &[_][]const u8{ "cat", "dog", "cat", "bird" };
    const codes = try encoder.fit_transform(labels);
    defer std.testing.allocator.free(codes);

    try std.testing.expectEqual(@as(usize, 3), encoder.classes_.len);
    try std.testing.expectEqual(@as(usize, 0), codes[0]);
    try std.testing.expectEqual(@as(usize, 1), codes[1]);
    try std.testing.expectEqual(@as(usize, 0), codes[2]);
    try std.testing.expectEqual(@as(usize, 2), codes[3]);

    const decoded = try encoder.inverse_transform(codes, std.testing.allocator);
    defer {
        for (decoded) |s| std.testing.allocator.free(s);
        std.testing.allocator.free(decoded);
    }
    try std.testing.expectEqualStrings("cat", decoded[0]);
    try std.testing.expectEqualStrings("dog", decoded[1]);
    try std.testing.expectEqualStrings("bird", decoded[3]);
}

test "OrdinalEncoder round trip" {
    var encoder = OrdinalEncoder.init(std.testing.allocator);
    defer encoder.deinit();

    const columns = &[_][]const []const u8{
        &[_][]const u8{ "a", "b", "a", "c" },
        &[_][]const u8{ "x", "y", "y", "x" },
    };
    var codes = try encoder.fit_transform(columns);
    defer codes.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 4), codes.rows);
    try std.testing.expectEqual(@as(usize, 2), codes.cols);
    try std.testing.expectEqual(@as(usize, 0), try codes.get(0, 0));
    try std.testing.expectEqual(@as(usize, 1), try codes.get(1, 0));
    try std.testing.expectEqual(@as(usize, 0), try codes.get(0, 1));

    const decoded = try encoder.inverse_transform(codes, std.testing.allocator);
    defer {
        for (decoded) |col| {
            for (col) |s| std.testing.allocator.free(s);
            std.testing.allocator.free(col);
        }
        std.testing.allocator.free(decoded);
    }
    try std.testing.expectEqualStrings("a", decoded[0][0]);
    try std.testing.expectEqualStrings("b", decoded[0][1]);
}

test "OneHotEncoder basic" {
    var encoder = OneHotEncoder.init(std.testing.allocator);
    defer encoder.deinit();

    const columns = &[_][]const []const u8{
        &[_][]const u8{ "a", "b", "a" },
        &[_][]const u8{ "x", "y", "y" },
    };
    var encoded = try encoder.fit_transform(columns);
    defer encoded.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 3), encoded.rows);
    try std.testing.expectEqual(@as(usize, 4), encoded.cols);
    try std.testing.expectEqual(@as(f64, 1.0), try encoded.get(0, 0));
    try std.testing.expectEqual(@as(f64, 1.0), try encoded.get(0, 2));
    try std.testing.expectEqual(@as(f64, 1.0), try encoded.get(1, 1));
    try std.testing.expectEqual(@as(f64, 1.0), try encoded.get(2, 3));
}
