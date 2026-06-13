const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    {
        var c_count = try zsl.iter.Comb.init(4, 2);
        defer c_count.deinit();
        std.debug.print("Combinations C(4, 2): count={d}\n", .{c_count.count()});
    }
    {
        var c = try zsl.iter.Comb.init(4, 2);
        defer c.deinit();
        while (c.next()) |comb| {
            std.debug.print("  {any}\n", .{comb});
        }
    }

    std.debug.print("\nPermutations of indices (3):\n", .{});
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

    std.debug.print("\nCounter start=1.5 step=0.5 (first 5):\n", .{});
    {
        var it = zsl.iter.counter(f64, 1.5, 0.5);
        for (0..5) |_| {
            std.debug.print("  {d}\n", .{it.next().?});
        }
    }

    std.debug.print("\nCycler over {{10, 20, 30}} (first 7):\n", .{});
    {
        var it = zsl.iter.cycler(i32, &[_]i32{ 10, 20, 30 });
        for (0..7) |_| {
            std.debug.print("  {d}\n", .{it.next().?});
        }
    }

    std.debug.print("\nRepeater value=7 (first 4):\n", .{});
    {
        var it = zsl.iter.repeater(i32, 7);
        for (0..4) |_| {
            std.debug.print("  {d}\n", .{it.next().?});
        }
    }

    std.debug.print("\nIntIter 0..10 step 3:\n", .{});
    {
        var it = try zsl.iter.int_iter(0, 10, 3);
        while (it.next()) |v| {
            std.debug.print("  {d}\n", .{v});
        }
    }

    std.debug.print("\nFloatIter 0.0..1.0 step 0.25:\n", .{});
    {
        var it = try zsl.iter.float_iter(0.0, 1.0, 0.25);
        while (it.next()) |v| {
            std.debug.print("  {d}\n", .{v});
        }
    }

    std.debug.print("\nPermutations of slice {{1, 2, 3}}:\n", .{});
    {
        var it = try zsl.iter.Permutations(i32).init(allocator, &[_]i32{ 1, 2, 3 });
        defer it.deinit(allocator);
        std.debug.print("count={d}\n", .{it.count()});
        while (it.next()) |perm| {
            std.debug.print("  {any}\n", .{perm});
        }
    }

    std.debug.print("\nEager permutations of {{a, b}}:\n", .{});
    {
        const perms = try zsl.iter.permutations(u8, allocator, &[_]u8{ 'a', 'b' });
        defer zsl.iter.permutations_free(u8, allocator, perms);
        for (perms) |perm| {
            std.debug.print("  {s}\n", .{perm});
        }
    }

    std.debug.print("\nProduct of {{1, 2}} x {{10, 20, 30}}:\n", .{});
    {
        var it = try zsl.iter.Product(i32).init(&[_]i32{ 1, 2 }, &[_]i32{ 10, 20, 30 });
        while (it.next()) |pair| {
            std.debug.print("  {any}\n", .{pair});
        }
    }

    std.debug.print("\nEager product of {{1, 2}} x {{10, 20}}:\n", .{});
    {
        const prod = try zsl.iter.product(i32, allocator, &[_]i32{ 1, 2 }, &[_]i32{ 10, 20 });
        defer zsl.iter.product_free(i32, allocator, prod);
        for (prod) |pair| {
            std.debug.print("  {any}\n", .{pair});
        }
    }
}
