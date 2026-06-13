const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    std.debug.print("gamma(5)      = {d}\n", .{zsl.fun.gamma.gamma(5.0)});
    std.debug.print("gamma(0.5)    = {d}\n", .{zsl.fun.gamma.gamma(0.5)});
    std.debug.print("digamma(1)    = {d}\n", .{zsl.fun.digamma.digamma(1.0)});
    std.debug.print("erf(1)        = {d}\n", .{zsl.fun.erf.erf(1.0)});
    std.debug.print("j0(1)         = {d}\n", .{zsl.fun.bessel.j0(1.0)});
    std.debug.print("i0(1)         = {d}\n", .{zsl.fun.mod_bessel.i0(1.0)});
    std.debug.print("choose(5, 2)  = {d}\n", .{try zsl.fun.misc.choose(5, 2)});
    std.debug.print("fib(10)       = {d}\n", .{try zsl.fun.misc.fib(10)});
    std.debug.print("hypot(3, 4)   = {d}\n", .{zsl.fun.misc.hypot(3.0, 4.0)});
    std.debug.print("cheb_eval([1,0,2], 0.5) = {d}\n", .{zsl.fun.interp.cheb_eval(&[_]f64{ 1.0, 0.0, 2.0 }, 0.5)});
}
