const std = @import("std");
const zsl = @import("zsl");
const poly = zsl.poly;

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    // P(x) = x^2 - 2x + 1
    const c = &[_]f64{ 1.0, -2.0, 1.0 };
    const x: f64 = 3.0;

    std.debug.print("P(x) = x^2 - 2x + 1\n", .{});
    std.debug.print("P({d}) = {d}\n", .{ x, try poly.eval(c, x) });

    const derivs = try poly.eval_derivs(c, x, 3, allocator);
    defer allocator.free(derivs);
    std.debug.print("P({d}) = {d}, P'({d}) = {d}, P''({d}) = {d}\n", .{
        x, derivs[0], x, derivs[1], x, derivs[2],
    });

    const quad_roots = try poly.solve_quadratic(1.0, -5.0, 6.0, allocator);
    defer allocator.free(quad_roots);
    std.debug.print("roots of x^2 - 5x + 6 = {any}\n", .{quad_roots});

    const cubic_roots = try poly.solve_cubic(1.0, -6.0, 11.0, -6.0, allocator);
    defer allocator.free(cubic_roots);
    std.debug.print("roots of x^3 - 6x^2 + 11x - 6 = {any}\n", .{cubic_roots});
}
