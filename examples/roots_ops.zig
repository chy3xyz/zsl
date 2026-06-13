const std = @import("std");
const zsl = @import("zsl");

fn f1(x: f64) f64 {
    return x * x - 4.0;
}

fn df1(x: f64) f64 {
    return 2.0 * x;
}

fn f2(x: f64) f64 {
    return x * x * x - 2.0;
}

fn df2(x: f64) f64 {
    return 3.0 * x * x;
}

pub fn main() !void {
    const tol = 1e-10;
    const max_iter = 100;

    const r_bisection = try zsl.roots.bisection(f1, 0.0, 3.0, tol, max_iter);
    std.debug.print("bisection(x^2 - 4)  = {d}\n", .{r_bisection});

    const r_brent = try zsl.roots.brent(f1, 0.0, 3.0, tol, max_iter);
    std.debug.print("brent(x^2 - 4)      = {d}\n", .{r_brent});

    const r_newton = try zsl.roots.newton(f1, df1, 3.0, tol, max_iter);
    std.debug.print("newton(x^2 - 4)     = {d}\n", .{r_newton});

    const r_newton_bisection = try zsl.roots.newton_bisection(f1, df1, 0.0, 3.0, tol, max_iter);
    std.debug.print("newton_bisection(x^2 - 4) = {d}\n", .{r_newton_bisection});

    const r_cube = try zsl.roots.bisection(f2, 0.0, 2.0, tol, max_iter);
    std.debug.print("bisection(x^3 - 2)  = {d}\n", .{r_cube});
}
