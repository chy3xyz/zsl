const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const data = &[_]f64{ 0.5, 1.5, 2.5, 3.5, 4.5 };
    const bins: usize = 4;

    const result = try zsl.dist.hist(data, bins, allocator);
    defer result.deinit(allocator);

    std.debug.print("Histogram of {any} with {d} bins\n", .{ data, bins });
    for (result.counts, 0..) |c, i| {
        std.debug.print("bin {d}: [{d:.4}, {d:.4}) -> {d}\n", .{
            i,
            result.edges[i],
            result.edges[i + 1],
            c,
        });
    }
}
