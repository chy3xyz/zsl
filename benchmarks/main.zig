const std = @import("std");
const blas_bench = @import("blas_bench.zig");
const lapack_bench = @import("lapack_bench.zig");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    std.debug.print("zsl benchmark harness\n", .{});

    try blas_bench.runAll(allocator);
    try lapack_bench.runAll(allocator);

    std.debug.print("\nBenchmarks complete.\n", .{});
}
