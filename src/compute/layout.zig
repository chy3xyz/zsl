const std = @import("std");
const Error = @import("../errors.zig").Error;

/// Convert a row-major `rows x cols` matrix slice to column-major order.
/// Caller owns the returned slice.
pub fn rowMajorToColumnMajor(
    allocator: std.mem.Allocator,
    src: []const f64,
    rows: usize,
    cols: usize,
) Error![]f64 {
    if (rows == 0 or cols == 0) return error.InvalidDimension;
    if (src.len != rows * cols) return error.ShapeMismatch;

    const dst = try allocator.alloc(f64, src.len);
    errdefer allocator.free(dst);

    for (0..rows) |i| {
        for (0..cols) |j| {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
    return dst;
}

/// Convert a column-major `rows x cols` matrix slice to row-major order.
/// Caller owns the returned slice.
pub fn columnMajorToRowMajor(
    allocator: std.mem.Allocator,
    src: []const f64,
    rows: usize,
    cols: usize,
) Error![]f64 {
    if (rows == 0 or cols == 0) return error.InvalidDimension;
    if (src.len != rows * cols) return error.ShapeMismatch;

    const dst = try allocator.alloc(f64, src.len);
    errdefer allocator.free(dst);

    for (0..rows) |i| {
        for (0..cols) |j| {
            dst[i * cols + j] = src[j * rows + i];
        }
    }
    return dst;
}

test "row-major to column-major round trip" {
    const src = &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const col = try rowMajorToColumnMajor(std.testing.allocator, src, 2, 3);
    defer std.testing.allocator.free(col);
    const dst = try columnMajorToRowMajor(std.testing.allocator, col, 2, 3);
    defer std.testing.allocator.free(dst);
    try std.testing.expectEqualSlices(f64, src, dst);
}

test "layout helpers reject invalid dimensions" {
    const src = &[_]f64{1.0};
    try std.testing.expectError(error.InvalidDimension, rowMajorToColumnMajor(std.testing.allocator, src, 0, 1));
    try std.testing.expectError(error.ShapeMismatch, rowMajorToColumnMajor(std.testing.allocator, src, 2, 2));
}
