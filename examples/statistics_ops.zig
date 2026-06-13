const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const T = f64;
    const M = zsl.la.Matrix(T);
    const stats = zsl.la.statistics;

    var data = try M.fromRowSlice(allocator, 4, 2, &[_]T{
        1.0, 2.0,
        2.0, 1.0,
        3.0, 4.0,
        4.0, 3.0,
    });
    defer data.deinit(allocator);

    var corr = try stats.correlation_matrix(data, allocator);
    defer corr.deinit(allocator);
    std.debug.print("correlation_matrix = {any}\n", .{corr.rawData()});

    var cov = try stats.covariance_matrix(data, 1, allocator);
    defer cov.deinit(allocator);
    std.debug.print("covariance_matrix (sample) = {any}\n", .{cov.rawData()});

    var centered = try stats.center_matrix(data, allocator);
    defer centered.deinit(allocator);
    std.debug.print("center_matrix = {any}\n", .{centered.rawData()});

    var standardized = try stats.standardize_matrix(data, allocator);
    defer standardized.deinit(allocator);
    std.debug.print("standardize_matrix = {any}\n", .{standardized.rawData()});
}
