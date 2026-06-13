const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var dispatch = try zsl.compute.ComputeDispatch.init(allocator, .cpu);
    defer dispatch.deinit();

    const a = &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const b = &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var c = [_]f64{ 0.0, 0.0, 0.0, 0.0 };

    try dispatch.gemm(a, b, &c, 2, 2, 3);
    std.debug.print("CPU gemm result: {any:.2}\n", .{c});

    var x = [_]f64{ 1.0, 2.0, 3.0 };
    try dispatch.softmax(&x, 1, 3);
    std.debug.print("CPU softmax result: {any:.4}\n", .{x});
}
