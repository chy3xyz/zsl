const std = @import("std");
const zsl = @import("zsl");
const gm = zsl.gm;

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    var bins = try gm.Bins.init(
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 10.0, 10.0 },
        &[_]usize{ 2, 2 },
        allocator,
    );
    defer bins.deinit(allocator);

    try bins.append(&[_]f64{ 1.0, 1.0 }, 10, null);
    try bins.append(&[_]f64{ 1.5, 1.5 }, 20, null);
    try bins.append(&[_]f64{ 6.0, 6.0 }, 30, null);

    std.debug.print("Bins created with ndim={d}\n", .{bins.ndim});

    if (try bins.find(&[_]f64{ 1.1, 1.1 }, 0.5)) |entry| {
        std.debug.print("find(1.1, 1.1) -> id={d}\n", .{entry.id});
    } else {
        std.debug.print("find(1.1, 1.1) -> not found\n", .{});
    }

    const all = try bins.find_all(&[_]f64{ 1.1, 1.1 }, 1.0, allocator);
    defer allocator.free(all);
    std.debug.print("find_all(1.1, 1.1) -> {d} entries:\n", .{all.len});
    for (all) |entry| {
        std.debug.print("  id={d}, x={any}\n", .{ entry.id, entry.x });
    }

    if (try bins.find(&[_]f64{ 11.0, 5.0 }, 0.5)) |_| {
        std.debug.print("out-of-range point was found (unexpected)\n", .{});
    } else {
        std.debug.print("out-of-range point returned null\n", .{});
    }
}
