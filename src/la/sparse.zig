const std = @import("std");
const la = @import("../la.zig");
const util = @import("../util.zig");
const Error = @import("../errors.zig").Error;

/// Triplet represents one non-zero entry of a sparse matrix in COO format.
pub fn Triplet(comptime T: type) type {
    return struct {
        row: usize,
        col: usize,
        val: T,
    };
}

/// SparseMatrix stores a matrix in coordinate-list (COO) format.
pub fn SparseMatrix(comptime T: type) type {
    _ = util.Float(T);

    return struct {
        rows: usize,
        cols: usize,
        allocator: std.mem.Allocator,
        data: std.ArrayList(Triplet(T)),

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) Error!Self {
            if (rows == 0 or cols == 0) return error.InvalidDimension;
            return .{
                .rows = rows,
                .cols = cols,
                .allocator = allocator,
                .data = std.ArrayList(Triplet(T)).empty,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.data.deinit(self.allocator);
            self.rows = 0;
            self.cols = 0;
        }

        /// Insert or update the value at (row, col).
        pub fn put(self: *Self, row: usize, col: usize, val: T) Error!void {
            try util.checkIndex(self.rows, row);
            try util.checkIndex(self.cols, col);
            for (self.data.items) |*t| {
                if (t.row == row and t.col == col) {
                    t.val = val;
                    return;
                }
            }
            try self.data.append(self.allocator, .{ .row = row, .col = col, .val = val });
        }

        /// Return the value at (row, col) or null if the entry has not been set.
        pub fn get(self: Self, row: usize, col: usize) Error!?T {
            try util.checkIndex(self.rows, row);
            try util.checkIndex(self.cols, col);
            for (self.data.items) |t| {
                if (t.row == row and t.col == col) return t.val;
            }
            return null;
        }

        /// Convert this sparse matrix to a dense row-major matrix.
        pub fn to_dense(self: Self, allocator: std.mem.Allocator) Error!la.Matrix(T) {
            var m = try la.Matrix(T).init(allocator, self.rows, self.cols);
            errdefer m.deinit(allocator);
            for (self.data.items) |t| {
                try m.set(t.row, t.col, t.val);
            }
            return m;
        }

        /// Sparse matrix-vector multiply y = A * x.
        pub fn spmv(self: Self, allocator: std.mem.Allocator, x: []const T) Error!la.Vector(T) {
            if (x.len != self.cols) return error.ShapeMismatch;
            var y = try la.Vector(T).init(allocator, self.rows);
            errdefer y.deinit(allocator);
            for (self.data.items) |t| {
                y.data[t.row] += t.val * x[t.col];
            }
            return y;
        }
    };
}

test "SparseMatrix put/get" {
    const T = f64;
    var sm = try SparseMatrix(T).init(std.testing.allocator, 3, 4);
    defer sm.deinit(std.testing.allocator);

    try sm.put(1, 2, 5.0);
    try std.testing.expectEqual(5.0, (try sm.get(1, 2)).?);
    try std.testing.expectEqual(null, try sm.get(0, 0));

    // Updating an existing entry should not add a duplicate triplet.
    try sm.put(1, 2, 7.0);
    try std.testing.expectEqual(7.0, (try sm.get(1, 2)).?);
    try std.testing.expectEqual(@as(usize, 1), sm.data.items.len);

    try std.testing.expectError(error.IndexOutOfBounds, sm.put(3, 0, 1.0));
    try std.testing.expectError(error.IndexOutOfBounds, sm.get(0, 4));
}

test "SparseMatrix to_dense" {
    const T = f64;
    var sm = try SparseMatrix(T).init(std.testing.allocator, 2, 3);
    defer sm.deinit(std.testing.allocator);

    try sm.put(0, 1, 2.0);
    try sm.put(1, 0, 3.0);
    try sm.put(1, 2, 4.0);

    var dense = try sm.to_dense(std.testing.allocator);
    defer dense.deinit(std.testing.allocator);

    try std.testing.expectEqual(0.0, try dense.get(0, 0));
    try std.testing.expectEqual(2.0, try dense.get(0, 1));
    try std.testing.expectEqual(0.0, try dense.get(0, 2));
    try std.testing.expectEqual(3.0, try dense.get(1, 0));
    try std.testing.expectEqual(0.0, try dense.get(1, 1));
    try std.testing.expectEqual(4.0, try dense.get(1, 2));
}

test "SparseMatrix spmv" {
    const T = f64;
    var sm = try SparseMatrix(T).init(std.testing.allocator, 2, 3);
    defer sm.deinit(std.testing.allocator);

    try sm.put(0, 0, 1.0);
    try sm.put(0, 2, 2.0);
    try sm.put(1, 1, 3.0);

    const x = &[_]T{ 4.0, 5.0, 6.0 };
    var y = try sm.spmv(std.testing.allocator, x);
    defer y.deinit(std.testing.allocator);

    try std.testing.expectApproxEqAbs(16.0, try y.get(0), 1e-12); // 1*4 + 2*6
    try std.testing.expectApproxEqAbs(15.0, try y.get(1), 1e-12); // 3*5
}

test "SparseMatrix spmv shape mismatch" {
    const T = f64;
    var sm = try SparseMatrix(T).init(std.testing.allocator, 2, 3);
    defer sm.deinit(std.testing.allocator);

    const x = &[_]T{ 1.0, 2.0 };
    try std.testing.expectError(error.ShapeMismatch, sm.spmv(std.testing.allocator, x));
}
