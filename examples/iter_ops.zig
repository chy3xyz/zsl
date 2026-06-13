const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    std.debug.print("Combinations C(4, 2): count={d}\n", .{(try zsl.iter.Comb.init(4, 2)).count()});
    {
        var c = try zsl.iter.Comb.init(4, 2);
        defer c.deinit();
        while (c.next()) |comb| {
            std.debug.print("  {any}\n", .{comb});
        }
    }

    std.debug.print("\nPermutations of 3:\n", .{});
    {
        var p = try zsl.iter.Perm.init(3, allocator);
        defer p.deinit(allocator);
        std.debug.print("count={d}\n", .{p.count()});
        while (p.next()) |perm| {
            std.debug.print("  {any}\n", .{perm});
        }
    }

    std.debug.print("\nCartesian product of dimensions {{2, 3}}:\n", .{});
    {
        var p = try zsl.iter.Prod.init(&[_]usize{ 2, 3 }, allocator);
        defer p.deinit(allocator);
        std.debug.print("count={d}\n", .{p.count()});
        while (p.next()) |prod| {
            std.debug.print("  {any}\n", .{prod});
        }
    }

    std.debug.print("\nRange 0..10 step 3:\n", .{});
    {
        var r = try zsl.iter.Ranges.init(0, 10, 3);
        while (r.next()) |v| {
            std.debug.print("  {d}\n", .{v});
        }
    }

    std.debug.print("\nInfIters start=5 step=2 (first 6):\n", .{});
    {
        var it = zsl.iter.InfIters.init(5, 2);
        for (0..6) |_| {
            std.debug.print("  {d}\n", .{it.next()});
        }
    }
}
